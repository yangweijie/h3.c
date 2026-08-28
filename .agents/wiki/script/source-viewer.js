/* ==========================================================================
   code-wiki :: source viewer core
   --------------------------------------------------------------------------
   Resolves source files through two channels and renders them with syntax
   highlighting, line numbers and "snippet / full file" switching.

     channel 1 (live)     GET <root>__source__?path=...   -> served by
                          `render.py serve`; any file in the repo, always
                          up to date. Requires http(s).
     channel 2 (embedded) <root>data/src/<id>.js          -> JSONP-style
                          payload emitted at render time for every file the
                          wiki references. Works on file:// and on any plain
                          static server.

   Globals exported: window.CWSrc
   ========================================================================== */
(function (global) {
  'use strict';

  var ROOT = global.__WIKI_ROOT__ || './';
  var LIVE = /^https?:$/.test(location.protocol);
  var liveDead = false;              // flip once /__source__ proves absent

  /* ----------------------------------------------------------- utilities */

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  var EXT = {
    js: 'javascript', mjs: 'javascript', cjs: 'javascript', jsx: 'javascript',
    ts: 'typescript', tsx: 'typescript', mts: 'typescript',
    py: 'python', pyw: 'python', pyi: 'python',
    rb: 'ruby', go: 'go', rs: 'rust', java: 'java', kt: 'kotlin', kts: 'kotlin',
    swift: 'swift', scala: 'scala', dart: 'dart', lua: 'lua', pl: 'perl', pm: 'perl',
    php: 'php', c: 'c', h: 'c', cc: 'cpp', cpp: 'cpp', cxx: 'cpp', hpp: 'cpp', hh: 'cpp',
    cs: 'csharp', m: 'objectivec', mm: 'objectivec', r: 'r',
    sh: 'bash', bash: 'bash', zsh: 'bash', fish: 'bash', ps1: 'powershell', psm1: 'powershell',
    sql: 'sql', html: 'xml', htm: 'xml', xml: 'xml', svg: 'xml', vue: 'xml', xhtml: 'xml',
    css: 'css', scss: 'scss', sass: 'scss', less: 'less',
    json: 'json', jsonc: 'json', yml: 'yaml', yaml: 'yaml',
    toml: 'ini', ini: 'ini', cfg: 'ini', conf: 'ini', properties: 'ini', env: 'ini',
    md: 'markdown', markdown: 'markdown', mdx: 'markdown',
    graphql: 'graphql', gql: 'graphql', proto: 'protobuf',
    diff: 'diff', patch: 'diff', vb: 'vbnet', wat: 'wasm', tf: 'ini', gradle: 'java'
  };
  var BYNAME = {
    dockerfile: 'dockerfile', makefile: 'makefile', 'cmakelists.txt': 'cmake',
    'gemfile': 'ruby', 'rakefile': 'ruby', '.gitignore': 'bash', '.env': 'ini',
    'nginx.conf': 'nginx', 'go.mod': 'ini', 'go.sum': 'ini'
  };

  function langOf(path, hint) {
    if (hint) return hint;
    var base = String(path).split('/').pop().toLowerCase();
    if (BYNAME[base]) return BYNAME[base];
    if (base.indexOf('dockerfile') === 0) return 'dockerfile';
    var i = base.lastIndexOf('.');
    return i < 0 ? '' : (EXT[base.slice(i + 1)] || '');
  }

  /* ------------------------------------------------------- source loading */

  var cache = {};
  var waiting = {};

  // called by data/src/<id>.js payloads
  global.__wikiSrcRecv = function (rec) {
    rec.kind = rec.kind || 'file';
    cache[rec.path] = rec;
    var q = waiting[rec.path];
    if (q) { delete waiting[rec.path]; q.forEach(function (f) { f(rec); }); }
  };

  function loadEmbedded(path) {
    return new Promise(function (resolve, reject) {
      if (cache[path]) return resolve(cache[path]);
      var idx = global.__WIKI_SRC_INDEX__ || {};
      var meta = idx[path];
      if (!meta) return reject(new Error('NOT_INDEXED'));
      (waiting[path] = waiting[path] || []).push(resolve);
      var s = document.createElement('script');
      s.src = ROOT + 'data/src/' + meta.id + '.js';
      s.onerror = function () { delete waiting[path]; reject(new Error('LOAD_FAIL')); };
      document.head.appendChild(s);
    });
  }

  function loadLive(path) {
    return fetch(ROOT + '__source__?path=' + encodeURIComponent(path))
      .then(function (r) {
        if (r.status === 404 || r.status === 501) { liveDead = true; throw new Error('NO_SERVER'); }
        if (!r.ok) throw new Error('HTTP ' + r.status);
        var ct = r.headers.get('content-type') || '';
        if (ct.indexOf('json') < 0) { liveDead = true; throw new Error('NO_SERVER'); }
        return r.json();
      })
      .then(function (j) {
        if (j.error) throw new Error(j.error);
        j.live = true;
        return j;
      });
  }

  function get(path) {
    if (cache[path]) return Promise.resolve(cache[path]);
    var p = (LIVE && !liveDead)
      ? loadLive(path).catch(function (e) {
          if (e && e.message === 'NOT_FOUND') throw e;
          return loadEmbedded(path);
        })
      : loadEmbedded(path);
    return p.then(function (rec) {
      rec.kind = rec.kind || 'file';
      rec.lang = langOf(rec.path || path, rec.lang);
      cache[rec.path || path] = rec;
      return rec;
    });
  }

  /* ---------------------------------------------------------- highlighting */

  // Re-open highlight.js spans that straddle a newline so every line can live
  // in its own DOM row (required for the line-number gutter).
  function splitLines(h) {
    var raw = h.split('\n'), stack = [], out = [];
    for (var i = 0; i < raw.length; i++) {
      var line = raw[i], local = stack.slice(), m;
      var re = /<(\/?)span([^>]*?)>/g;
      while ((m = re.exec(line))) {
        if (m[1] === '/') local.pop(); else local.push('<span' + m[2] + '>');
      }
      out.push(stack.join('') + line + new Array(local.length + 1).join('</span>'));
      stack = local;
    }
    return out;
  }

  function highlight(text, lang) {
    var h;
    if (global.hljs && text.length < 900000) {
      try {
        if (lang && hljs.getLanguage(lang)) {
          h = hljs.highlight(text, { language: lang, ignoreIllegals: true }).value;
        } else if (text.length < 160000) {
          h = hljs.highlightAuto(text).value;
        } else { h = esc(text); }
      } catch (e) { h = esc(text); }
    } else { h = esc(text); }
    return splitLines(h);
  }

  /* -------------------------------------------------------------- rendering */

  var CONTEXT = 12;
  var MAX_ROWS = 24000;

  function chunkLines(rec) {
    // -> [{no:Number, html:String}]
    var rows = [];
    (rec.chunks || []).forEach(function (ch) {
      var text = ch.text == null ? '' : ch.text;
      if (text.charAt(text.length - 1) === '\n') text = text.slice(0, -1);
      var lines = highlight(text, rec.lang);
      for (var i = 0; i < lines.length; i++) rows.push({ no: ch.start + i, html: lines[i] });
    });
    return rows;
  }

  function rawLines(rec) {
    var rows = [];
    (rec.chunks || []).forEach(function (ch) {
      var text = ch.text == null ? '' : ch.text;
      if (text.charAt(text.length - 1) === '\n') text = text.slice(0, -1);
      var lines = text.split('\n');
      for (var i = 0; i < lines.length; i++) rows.push({ no: ch.start + i, text: lines[i] });
    });
    return rows;
  }

  function buildCode(rec, opt) {
    var lo = 1, hi = Infinity;
    if (opt.mode === 'part' && opt.start) {
      lo = Math.max(1, opt.start - CONTEXT);
      hi = (opt.end || opt.start) + CONTEXT;
    }
    var rows = chunkLines(rec);
    var html = [], last = 0, shown = 0, truncated = false;
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      if (r.no < lo || r.no > hi) continue;
      if (shown >= MAX_ROWS) { truncated = true; break; }
      if (last && r.no > last + 1) html.push(gapRow(last + 1, r.no - 1));
      var hl = (opt.start && r.no >= opt.start && r.no <= (opt.end || opt.start));
      html.push('<div class="cl' + (hl ? ' hl' : '') +
        (r.no === opt.start ? ' anchor' : '') + '" id="L' + r.no + '">' +
        '<a class="cn" href="#L' + r.no + '">' + r.no + '</a>' +
        '<span class="cc">' + (r.html || ' ') + '</span></div>');
      last = r.no; shown++;
    }
    if (!shown) return '<div class="sv-empty">该范围内没有可显示的内容。</div>';
    if (last && rec.total && last < rec.total && opt.mode !== 'part') {
      html.push(gapRow(last + 1, rec.total));
    }
    if (truncated) {
      html.push('<div class="gap">文件过大，仅渲染前 ' + MAX_ROWS + ' 行</div>');
    }
    return '<div class="code mono">' + html.join('') + '</div>';
  }

  function gapRow(a, b) {
    var n = b - a + 1;
    return '<div class="gap">⋯ 省略 ' + n + ' 行 (' + a + '–' + b + ') ' +
      '<button class="btn" data-act="full">显示完整文件</button></div>';
  }

  function buildDir(rec) {
    var out = ['<div class="sv-dir">'];
    (rec.entries || []).forEach(function (e) {
      out.push('<a href="#" data-open="' + esc(e.path) + '">' +
        (e.dir ? '📁 ' : '📄 ') + esc(e.name) + '</a>');
    });
    if (!(rec.entries || []).length) out.push('<span class="sv-empty">空目录</span>');
    out.push('</div>');
    return out.join('');
  }

  /* -------------------------------------------------------- viewer surface */

  // A "surface" is any {head, path, badge, note, body, tools} element set.
  function Surface(els) {
    this.els = els;
    this.state = { path: '', start: null, end: null, mode: 'part', rec: null };
    var self = this;
    els.body.addEventListener('click', function (ev) {
      var t = ev.target.closest ? ev.target.closest('[data-act],[data-open]') : null;
      if (!t) return;
      if (t.getAttribute('data-act') === 'full') { ev.preventDefault(); self.setMode('full'); }
      var op = t.getAttribute('data-open');
      if (op) { ev.preventDefault(); self.open(op, null, null, 'full'); }
    });
  }

  Surface.prototype.setMode = function (mode) {
    this.state.mode = mode;
    this.paint(true);
  };

  Surface.prototype.open = function (path, start, end, mode) {
    var self = this;
    this.state.path = path;
    this.state.start = start ? parseInt(start, 10) : null;
    this.state.end = end ? parseInt(end, 10) : null;
    this.state.mode = mode || (start ? 'part' : 'full');
    this.state.rec = null;
    this.paintHead();
    this.els.body.innerHTML = '<div class="sv-empty">加载中…</div>';
    get(path).then(function (rec) {
      self.state.rec = rec;
      self.paint(false);
    }).catch(function (err) {
      self.els.body.innerHTML = '<div class="sv-empty">' + failMsg(path, err) + '</div>';
    });
  };

  Surface.prototype.paintHead = function () {
    var s = this.state;
    var parts = s.path.split('/');
    var name = parts.pop();
    if (this.els.path) {
      this.els.path.innerHTML = '<span>' + esc(parts.length ? parts.join('/') + '/' : '') +
        '</span><b>' + esc(name) + '</b>';
      this.els.path.title = s.path;
    }
    if (this.els.badge) {
      this.els.badge.textContent = s.start
        ? ('L' + s.start + (s.end && s.end !== s.start ? '–' + s.end : ''))
        : '';
      this.els.badge.style.display = s.start ? '' : 'none';
    }
    if (this.els.btnPart) this.els.btnPart.classList.toggle('on', s.mode === 'part');
    if (this.els.btnFull) this.els.btnFull.classList.toggle('on', s.mode === 'full');
    if (this.els.btnPart) this.els.btnPart.disabled = !s.start;
  };

  Surface.prototype.paint = function (keepScroll) {
    var s = this.state, rec = s.rec;
    this.paintHead();
    if (!rec) return;
    if (rec.kind === 'dir') {
      this.note('');
      this.els.body.innerHTML = buildDir(rec);
      return;
    }
    var notes = [];
    if (rec.partial) {
      notes.push('该文件仅内嵌了被引用的片段' +
        (LIVE ? '' : '；用 <code>render.py serve</code> 启动本地服务后可查看完整文件') + '。');
    }
    if (rec.note) notes.push(esc(rec.note));
    if (rec.total) notes.push('共 ' + rec.total + ' 行' + (rec.lang ? ' · ' + rec.lang : ''));
    this.note(notes.join(' · '));
    this.els.body.innerHTML = buildCode(rec, s);
    if (!keepScroll) this.scrollToAnchor();
  };

  Surface.prototype.note = function (h) {
    if (!this.els.note) return;
    this.els.note.innerHTML = h || '';
    this.els.note.style.display = h ? '' : 'none';
  };

  Surface.prototype.scrollToAnchor = function () {
    var s = this.state;
    if (!s.start) { this.els.body.scrollTop = 0; return; }
    var el = this.els.body.querySelector('#L' + s.start);
    if (!el) { this.els.body.scrollTop = 0; return; }
    var top = el.offsetTop - this.els.body.clientHeight * 0.28;
    this.els.body.scrollTop = Math.max(0, top);
  };

  Surface.prototype.currentText = function () {
    var rec = this.state.rec;
    if (!rec || rec.kind === 'dir') return '';
    var s = this.state, lo = 1, hi = Infinity;
    if (s.mode === 'part' && s.start) {
      lo = Math.max(1, s.start - CONTEXT);
      hi = (s.end || s.start) + CONTEXT;
    }
    return rawLines(rec).filter(function (r) { return r.no >= lo && r.no <= hi; })
      .map(function (r) { return r.text; }).join('\n');
  };

  function failMsg(path, err) {
    var m = err && err.message;
    if (m === 'NOT_INDEXED') {
      return '未内嵌该文件：<code>' + esc(path) + '</code><br><br>' +
        '它没有被任何 wiki 页面引用。运行 <code>python render.py serve &lt;workspace&gt;</code> ' +
        '后即可浏览仓库内任意文件。';
    }
    if (m === 'NOT_FOUND') return '文件不存在：<code>' + esc(path) + '</code>';
    if (m === 'OUTSIDE') return '拒绝访问工作区之外的路径。';
    if (m === 'BINARY') return '二进制文件，无法以文本预览。';
    return '加载失败：' + esc(m || 'unknown') + '<br><code>' + esc(path) + '</code>';
  }

  /* ------------------------------------------------------------- exports */

  global.CWSrc = {
    ROOT: ROOT,
    LIVE: LIVE,
    esc: esc,
    langOf: langOf,
    get: get,
    highlight: highlight,
    Surface: Surface,
    isLive: function () { return LIVE && !liveDead; }
  };
})(window);
