#import "h3_ane_bridge.h"

#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

struct h3_ane_model {
    void *model;
    void *request;
    char *staging_directory;
    double compile_seconds;
    bool cache_hit;
};

static void bridge_fail(char *error, size_t error_size, const char *format,
                        ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double bridge_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

int h3_ane_bridge_available(void) {
    static int state = -1;
    if (state >= 0) return state;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
           "AppleNeuralEngine", RTLD_NOW);
    state = NSClassFromString(@"_ANEInMemoryModelDescriptor") != nil &&
            NSClassFromString(@"_ANEInMemoryModel") != nil &&
            NSClassFromString(@"_ANERequest") != nil &&
            NSClassFromString(@"_ANEIOSurfaceObject") != nil;
    return state;
}

IOSurfaceRef h3_ane_bridge_surface(size_t bytes) {
    size_t aligned = (bytes + 16383u) & ~(size_t)16383u;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned),
        (id)kIOSurfaceAllocSize: @(aligned),
        (id)kIOSurfacePixelFormat: @0});
}

bool h3_ane_cache_enabled(void) {
    const char *env = getenv("H3_ANE_CACHE");
    return !env || atoi(env) != 0;
}

static void bridge_write_sources(NSString *directory, NSData *program,
                                 NSData *weights) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files createDirectoryAtPath:
        [directory stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [program writeToFile:
        [directory stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [weights writeToFile:
        [directory stringByAppendingPathComponent:@"weights/weight.bin"]
        atomically:YES];
}

static NSString *bridge_cache_root(void) {
    return [NSTemporaryDirectory()
        stringByAppendingPathComponent:@"h3-ane-cache"];
}

static NSString *bridge_cache_entry(NSString *identifier) {
    return [bridge_cache_root() stringByAppendingPathComponent:identifier];
}

/* Mirror a directory tree file by file, hardlinking each one (same volume, so
 * links are free) and copying only as the fallback. A directory cannot be
 * hardlinked, and copying an entry would double the disk peak: an fp16 video
 * VAE shape is ~4.6 GiB and the staging directory has to stay on this volume. */
static bool bridge_mirror(NSString *from, NSString *to) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files removeItemAtPath:to error:nil];
    if (![files createDirectoryAtPath:to withIntermediateDirectories:YES
                           attributes:nil error:nil])
        return false;
    for (NSString *relative in [files subpathsOfDirectoryAtPath:from error:nil]) {
        NSString *source = [from stringByAppendingPathComponent:relative];
        NSString *target = [to stringByAppendingPathComponent:relative];
        BOOL is_directory = NO;
        if (![files fileExistsAtPath:source isDirectory:&is_directory]) return false;
        if (is_directory) {
            if (![files createDirectoryAtPath:target withIntermediateDirectories:YES
                                  attributes:nil error:nil]) return false;
            continue;
        }
        if ([files linkItemAtPath:source toPath:target error:nil]) continue;
        [files removeItemAtPath:target error:nil];
        if (![files copyItemAtPath:source toPath:target error:nil]) return false;
    }
    return true;
}

/* The model unload deletes its staging directory, so compiled artifacts are
 * preserved in a content-addressed entry next to it and restored on reuse. */
static bool bridge_cache_restore(NSString *identifier, NSString *directory) {
    NSString *entry = bridge_cache_entry(identifier);
    NSFileManager *files = [NSFileManager defaultManager];
    NSString *marker = [entry stringByAppendingPathComponent:@"compiled.ok"];
    if (![files fileExistsAtPath:marker]) return false;
    if (!bridge_mirror(entry, directory)) return false;
    [files removeItemAtPath:
        [directory stringByAppendingPathComponent:@"compiled.ok"] error:nil];
    /* Last-use time for the LRU trim: a touch, not a rewrite. */
    [files setAttributes:@{@"modificationDate": [NSDate date]}
            ofItemAtPath:marker error:nil];
    return true;
}

/* Byte cap for the compiled-artifact cache. An fp16 video VAE shape is
 * ~4.6 GiB of entries, and ANE loads start failing once the system volume
 * drops near a few GiB, so letting the cache grow without bound turns the
 * next cold start into a compile failure. H3_ANE_CACHE_MAX_MIB overrides. */
static uint64_t bridge_cache_limit(void) {
    const char *env = getenv("H3_ANE_CACHE_MAX_MIB");
    double mib = env ? atof(env) : 5120.0;
    if (mib < 0.0) mib = 0.0;
    return (uint64_t)(mib * 1024.0 * 1024.0);
}

/* How much of the volume must stay free for an ANE load to be trusted; the
 * failures measured on this machine started around 4.4 GiB free. The cap alone
 * cannot express that, because other writers share the container. */
static uint64_t bridge_cache_min_free(void) {
    const char *env = getenv("H3_ANE_CACHE_MIN_FREE_MIB");
    double mib = env ? atof(env) : 6144.0;
    if (mib < 0.0) mib = 0.0;
    return (uint64_t)(mib * 1024.0 * 1024.0);
}

static uint64_t bridge_volume_free(NSString *path) {
    NSDictionary *attributes = [[NSFileManager defaultManager]
        attributesOfFileSystemForPath:path error:nil];
    return [[attributes objectForKey:NSFileSystemFreeSize]
        unsignedLongLongValue];
}

/* The smaller of the configured cap and whatever leaves the volume above the
 * free-space floor, assuming an evicted entry really returns its bytes. A live
 * graph still holding a hardlink delays that; the next trim tightens. */
static uint64_t bridge_cache_target(uint64_t total) {
    uint64_t limit = bridge_cache_limit();
    uint64_t floor = bridge_cache_min_free();
    if (!floor) return limit;
    uint64_t free_now = bridge_volume_free(bridge_cache_root());
    if (free_now <= floor) return total > (floor - free_now) ?
        total - (floor - free_now) : 0;
    uint64_t room = free_now - floor + total;
    return room < limit ? room : limit;
}

static uint64_t bridge_path_bytes(NSString *path) {
    NSFileManager *files = [NSFileManager defaultManager];
    uint64_t total = 0;
    for (NSString *relative in [files subpathsOfDirectoryAtPath:path error:nil]) {
        NSDictionary *attributes = [files attributesOfItemAtPath:
            [path stringByAppendingPathComponent:relative] error:nil];
        if (attributes && [[attributes fileType] isEqualToString:NSFileTypeRegular])
            total += [attributes fileSize];
    }
    return total;
}

/* Drop least-recently-used entries until the cache fits. The entry just
 * written (`keep`) never goes first, so a single shape that exceeds the cap
 * still gets one usable copy instead of thrashing on every graph. */
static void bridge_cache_trim(NSString *keep) {
    NSString *root = bridge_cache_root();
    NSFileManager *files = [NSFileManager defaultManager];
    /* An interrupted store leaves a .tmp behind, and once its staging links
     * are gone those are real bytes; the sweep is not gated on the cap. */
    for (NSString *name in [files contentsOfDirectoryAtPath:root error:nil]) {
        if ([name hasSuffix:@".tmp"])
            [files removeItemAtPath:[root stringByAppendingPathComponent:name]
                              error:nil];
    }
    NSMutableArray<NSString *> *entries = [NSMutableArray array];
    for (NSString *name in [files contentsOfDirectoryAtPath:root error:nil]) {
        NSString *entry = [root stringByAppendingPathComponent:name];
        if (![files fileExistsAtPath:
                [entry stringByAppendingPathComponent:@"compiled.ok"]]) continue;
        [entries addObject:entry];
    }
    uint64_t total = 0;
    for (NSString *entry in entries) total += bridge_path_bytes(entry);
    uint64_t target = bridge_cache_target(total);
    if (total <= target) return;
    [entries sortUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
        NSDate *ma = [[files attributesOfItemAtPath:
            [a stringByAppendingPathComponent:@"compiled.ok"] error:nil]
            fileModificationDate];
        NSDate *mb = [[files attributesOfItemAtPath:
            [b stringByAppendingPathComponent:@"compiled.ok"] error:nil]
            fileModificationDate];
        if (!ma) return NSOrderedDescending;
        if (!mb) return NSOrderedAscending;
        return [ma compare:mb];
    }];
    for (NSString *entry in entries) {
        if (total <= target) break;
        if (keep && [entry.lastPathComponent isEqualToString:keep.lastPathComponent])
            continue;
        uint64_t bytes = bridge_path_bytes(entry);
        if ([files removeItemAtPath:entry error:nil]) {
            total -= bytes;
            if (getenv("H3_ANE_CACHE_DEBUG"))
                fprintf(stderr, "ANE cache: evicted %s (%llu MiB, now %llu MiB "
                        "of %llu MiB, volume free %llu MiB)\n",
                        entry.lastPathComponent.UTF8String,
                        (unsigned long long)(bytes >> 20),
                        (unsigned long long)(total >> 20),
                        (unsigned long long)(target >> 20),
                        (unsigned long long)(bridge_volume_free(root) >> 20));
        }
    }
}

/* Only the compiled artifacts are kept: `data` embeds the constants, so the
 * weights copy and the MIL text are dead weight in an entry. */
static void bridge_cache_store(NSString *identifier, NSString *directory) {
    NSString *entry = bridge_cache_entry(identifier);
    NSString *staged = [entry stringByAppendingString:@".tmp"];
    NSFileManager *files = [NSFileManager defaultManager];
    /* Build the replacement beside the old entry and swap it in, so an
     * interrupted store can never leave a hole where a working artifact was. */
    [files removeItemAtPath:staged error:nil];
    if (![files createDirectoryAtPath:staged withIntermediateDirectories:YES
                           attributes:nil error:nil]) return;
    for (NSString *name in
         [files contentsOfDirectoryAtPath:directory error:nil]) {
        if ([name isEqualToString:@"weights"] ||
            [name isEqualToString:@"model.mil"]) continue;
        NSString *from = [directory stringByAppendingPathComponent:name];
        NSString *to = [staged stringByAppendingPathComponent:name];
        if (![files linkItemAtPath:from toPath:to error:nil]) {
            [files removeItemAtPath:to error:nil];
            if (![files copyItemAtPath:from toPath:to error:nil]) {
                [files removeItemAtPath:staged error:nil];
                return;
            }
        }
    }
    if (![[NSData data] writeToFile:
            [staged stringByAppendingPathComponent:@"compiled.ok"]
            atomically:YES]) {
        [files removeItemAtPath:staged error:nil];
        return;
    }
    [files removeItemAtPath:entry error:nil];
    NSError *failure = nil;
    if (![files moveItemAtPath:staged toPath:entry error:&failure]) {
        [files removeItemAtPath:staged error:nil];
        return;
    }
    bridge_cache_trim(entry);
}

static void bridge_cache_evict(const char *identifier) {
    if (!identifier) return;
    [[NSFileManager defaultManager]
        removeItemAtPath:bridge_cache_entry(@(identifier)) error:nil];
}

h3_ane_model *h3_ane_model_create(const char *name, const char *mil,
                                  void *weight_bytes_owned, size_t weight_bytes,
                                  IOSurfaceRef *input_surfaces,
                                  uint32_t input_count, IOSurfaceRef output,
                                  char *error, size_t error_size) {
    if (!h3_ane_bridge_available()) {
        free(weight_bytes_owned);
        bridge_fail(error, error_size, "the Neural Engine bridge is "
                    "unavailable");
        return NULL;
    }
    h3_ane_model *handle = calloc(1, sizeof(*handle));
    if (!handle) {
        free(weight_bytes_owned);
        bridge_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    @autoreleasepool {
        NSError *failure = nil;
        NSData *weights = [NSData dataWithBytesNoCopy:weight_bytes_owned
                                               length:weight_bytes
                                         freeWhenDone:YES];
        NSData *program = [NSData dataWithBytes:mil length:strlen(mil)];
        Class descriptorClass =
            NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            program, @{@"@model_path/weights/weight.bin":
                       @{@"offset": @0, @"data": weights}}, nil);
        if (!descriptor) {
            bridge_fail(error, error_size, "ANE %s descriptor rejected", name);
            free(handle);
            return NULL;
        }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!model) {
            bridge_fail(error, error_size, "ANE %s model rejected", name);
            free(handle);
            return NULL;
        }
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(
            model, @selector(hexStringIdentifier));
        NSString *directory = [NSTemporaryDirectory()
            stringByAppendingPathComponent:identifier];
        NSFileManager *files = [NSFileManager defaultManager];
        bool cache = h3_ane_cache_enabled();
        bool cached = cache && bridge_cache_restore(identifier, directory);
        if (getenv("H3_ANE_CACHE_DEBUG"))
            fprintf(stderr, "ANE %s: cache %s (%s)\n", name,
                    cached ? "restored" : (cache ? "miss" : "disabled"),
                    identifier.UTF8String);
        if (!cached && cache) {
            /* A cold compile needs volume headroom for the artifact it is about
             * to write, so make room first rather than fail at the end. */
            bridge_cache_trim(NULL);
        }
        if (!cached) bridge_write_sources(directory, program, weights);
        handle->staging_directory = strdup(directory.UTF8String);
        double started = bridge_seconds();
        bool loaded = cached &&
            ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{},
                &failure);
        if (cached && !loaded) {
            if (getenv("H3_ANE_CACHE_DEBUG"))
                fprintf(stderr, "ANE %s: cached load failed (%s); recompiling "
                        "from %s\n", name,
                        failure ? failure.localizedDescription.UTF8String : "?",
                        directory.UTF8String);
            failure = nil;
            bridge_write_sources(directory, program, weights);
        }
        if (!loaded) {
            if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(compileWithQoS:options:error:), 21, @{},
                    &failure)) {
                bridge_fail(error, error_size, "ANE %s compile failed: %s",
                            name, failure ?
                            failure.localizedDescription.UTF8String : "?");
                [files removeItemAtPath:directory error:nil];
                free(handle->staging_directory);
                free(handle);
                return NULL;
            }
            if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(loadWithQoS:options:error:), 21, @{},
                    &failure)) {
                bridge_fail(error, error_size, "ANE %s load failed: %s", name,
                            failure ?
                            failure.localizedDescription.UTF8String : "?");
                [files removeItemAtPath:directory error:nil];
                free(handle->staging_directory);
                free(handle);
                return NULL;
            }
            if (cache) bridge_cache_store(identifier, directory);
        }
        handle->cache_hit = loaded;
        handle->compile_seconds = bridge_seconds() - started;
        NSMutableArray *inputs = [NSMutableArray array];
        NSMutableArray *indices = [NSMutableArray array];
        for (uint32_t index = 0; index < input_count; index++) {
            [inputs addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:),
                input_surfaces[index])];
            [indices addObject:@(index)];
        }
        id wrapped = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), output);
        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))
                      objc_msgSend)(
            requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                      weightsBuffer:perfStats:procedureIndex:),
            inputs, indices, @[wrapped], @[@0], nil, nil, @0);
        if (!request) {
            bridge_fail(error, error_size, "ANE %s request rejected", name);
            h3_ane_model_free(handle);
            return NULL;
        }
        handle->model = (__bridge_retained void *)model;
        handle->request = (__bridge_retained void *)request;
    }
    return handle;
}

int h3_ane_model_eval(h3_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model || !handle->request) {
        bridge_fail(error, error_size, "the ANE model is not loaded");
        return 0;
    }
    int ok = 0;
    @autoreleasepool {
        NSError *failure = nil;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            (__bridge id)handle->model,
            @selector(evaluateWithQoS:options:request:error:), 21, @{},
            (__bridge id)handle->request, &failure) ? 1 : 0;
        if (!ok)
            bridge_fail(error, error_size, "ANE evaluation failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
    }
    return ok;
}

int h3_ane_model_unload(h3_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model) {
        bridge_fail(error, error_size, "no ANE model to unload");
        return 0;
    }
    @autoreleasepool {
        NSError *failure = nil;
        if (!((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                (__bridge id)handle->model, @selector(unloadWithQoS:error:), 21,
                &failure)) {
            bridge_fail(error, error_size, "ANE unload failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
            return 0;
        }
    }
    return 1;
}

int h3_ane_model_reload(h3_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model || !handle->staging_directory) {
        bridge_fail(error, error_size, "no ANE model to reload");
        return 0;
    }
    @autoreleasepool {
        NSString *directory = @(handle->staging_directory);
        bridge_cache_restore(directory.lastPathComponent, directory);
        NSError *failure = nil;
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                (__bridge id)handle->model,
                @selector(loadWithQoS:options:error:), 21, @{}, &failure)) {
            bridge_fail(error, error_size, "ANE reload failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
            return 0;
        }
    }
    return 1;
}

void h3_ane_model_free(h3_ane_model *handle) {
    if (!handle) return;
    @autoreleasepool {
        if (handle->model) {
            id model = (__bridge_transfer id)handle->model;
            NSError *failure = nil;
            ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &failure);
        }
        if (handle->request) {
            id request = (__bridge_transfer id)handle->request;
            (void)request;
        }
        if (handle->staging_directory) {
            [[NSFileManager defaultManager]
                removeItemAtPath:@(handle->staging_directory) error:nil];
            if (!h3_ane_cache_enabled())
                bridge_cache_evict(strrchr(handle->staging_directory, '/') ?
                    strrchr(handle->staging_directory, '/') + 1 :
                    handle->staging_directory);
            free(handle->staging_directory);
        }
    }
    free(handle);
}

double h3_ane_model_compile_seconds(const h3_ane_model *handle) {
    return handle ? handle->compile_seconds : 0.0;
}

bool h3_ane_model_cache_hit(const h3_ane_model *handle) {
    return handle ? handle->cache_hit : false;
}
