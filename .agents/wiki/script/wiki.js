/* ==========================================================================
   code-wiki :: page runtime
   Handles three contexts, auto-detected from the DOM:
     · pages/*.html  — article rendering, code highlighting, source drawer
     · index.html    — nav, search, theme toggle, iframe messaging
     · source.html   — standalone full-page source browser
   ========================================================================== */
(function () {
  'use strict';

  /* ------------------------------------------------------------- theme */

  var THEME_KEY = 'cw-theme';

  function readTheme() {
    try { return localStorage.getItem(THEME_KEY) || 'dark'; } catch (e) { return 'dark'; }
  }
  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t === 'light' ? 'light' : 'dark');
    reflowMermaid();
  }
  function saveTheme(t) { try { localStorage.setItem(THEME_KEY, t); } catch (e) {} }

  applyTheme(readTheme());

  window.addEventListener('message', function (ev) {
    var d = ev.data;
    if (d && d.cw === 'theme') applyTheme(d.theme);
  });

  /* ----------------------------------------------------------- mermaid */

  var mermaidSrc = [];

  function initMermaid() {
    if (!window.mermaid) return;
    var nodes = document.querySelectorAll('pre.mermaid');
    if (!nodes.length) return;
    for (var i = 0; i < nodes.length; i++) mermaidSrc.push(nodes[i].textContent);
    runMermaid();
  }

  function runMermaid() {
    if (!window.mermaid) return;
    var dark = document.documentElement.getAttribute('data-theme') !== 'light';
    try {
      mermaid.initialize({ startOnLoad: false, theme: dark ? 'dark' : 'default', securityLevel: 'loose' });
      var nodes = document.querySelectorAll('pre.mermaid');
      mermaid.run({ nodes: nodes });
    } catch (e) { /* ignore */ }
  }

  function reflowMermaid() {
    if (!window.mermaid || !mermaidSrc.length) return;
    var nodes = document.querySelectorAll('pre.mermaid');
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].removeAttribute('data-processed');
      nodes[i].textContent = mermaidSrc[i];
    }
    runMermaid();
  }

  /* -------------------------------------------------------- code blocks */

  function initCodeBlocks() {
    var blocks = document.querySelectorAll('.codeblock');
    for (var i = 0; i < blocks.length; i++) {
      (function (box) {
        var code = box.querySelector('code');
        if (code && window.hljs) {
          try { hljs.highlightElement(code); } catch (e) {}
        }
        var btn = box.querySelector('[data-cb="copy"]');
        if (btn && code) {
          btn.addEventListener('click', function () {
            copyText(code.textContent, btn);
          });
        }
      })(blocks[i]);
    }
  }

  function copyText(txt, btn) {
    var done = function () {
      if (!btn) return;
      var old = btn.textContent;
      btn.textContent = '已复制';
      setTimeout(function () { btn.textContent = old; }, 1200);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).then(done, function () { fallbackCopy(txt); done(); });
    } else { fallbackCopy(txt); done(); }
  }

  function fallbackCopy(txt) {
    var ta = document.createElement('textarea');
    ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
  }

  /* ------------------------------------------------------ source drawer */

  var drawer = null, surface = null;

  var DRAWER_HTML =
    '<div class="sv-backdrop"></div>' +
    '<aside class="sv-drawer">' +
      '<div class="sv-head">' +
        '<div class="sv-path"></div><span class="sv-badge"></span>' +
        '<div class="sv-tools">' +
          '<button class="btn" data-r="part" title="只看引用片段">片段</button>' +
          '<button class="btn" data-r="full" title="查看完整文件">完整</button>' +
          '<button class="btn" data-r="copy" title="复制当前可见代码">复制</button>' +
          '<button class="btn" data-r="wide" title="展开/收起宽度">⤢</button>' +
          '<button class="btn" data-r="new" title="在新标签页打开">↗</button>' +
          '<button class="btn" data-r="close" title="关闭 (Esc)">✕</button>' +
        '</div>' +
      '</div>' +
      '<div class="sv-note" style="display:none"></div>' +
      '<div class="sv-body"></div>' +
    '</aside>';

  function ensureDrawer() {
    if (drawer) return drawer;
    var host = document.createElement('div');
    host.innerHTML = DRAWER_HTML;
    while (host.firstChild) document.body.appendChild(host.firstChild);

    var back = document.querySelector('.sv-backdrop');
    var aside = document.querySelector('.sv-drawer');
    var els = {
      path: aside.querySelector('.sv-path'),
      badge: aside.querySelector('.sv-badge'),
      note: aside.querySelector('.sv-note'),
      body: aside.querySelector('.sv-body'),
      btnPart: aside.querySelector('[data-r="part"]'),
      btnFull: aside.querySelector('[data-r="full"]')
    };
    surface = new CWSrc.Surface(els);

    aside.querySelector('.sv-tools').addEventListener('click', function (ev) {
      var b = ev.target.closest('[data-r]');
      if (!b) return;
      var r = b.getAttribute('data-r');
      if (r === 'part' || r === 'full') surface.setMode(r);
      else if (r === 'copy') copyText(surface.currentText(), b);
      else if (r === 'wide') aside.classList.toggle('wide');
      else if (r === 'new') {
        var s = surface.state;
        var u = CWSrc.ROOT + 'source.html?f=' + encodeURIComponent(s.path) +
          (s.start ? '&s=' + s.start : '') + (s.end ? '&e=' + s.end : '') + '&mode=' + s.mode;
        window.open(u, '_blank');
      } else if (r === 'close') closeDrawer();
    });

    back.addEventListener('click', closeDrawer);
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') closeDrawer();
    });

    drawer = { aside: aside, back: back };
    return drawer;
  }

  function openDrawer(path, s, e) {
    var d = ensureDrawer();
    d.aside.classList.add('open');
    d.back.classList.add('open');
    surface.open(path, s, e, s ? 'part' : 'full');
  }

  function closeDrawer() {
    if (!drawer) return;
    drawer.aside.classList.remove('open');
    drawer.back.classList.remove('open');
  }

  function initSourceRefs() {
    document.addEventListener('click', function (ev) {
      var a = ev.target.closest ? ev.target.closest('a.src-ref') : null;
      if (!a) return;
      if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button === 1) return; // let it open a tab
      ev.preventDefault();
      openDrawer(a.getAttribute('data-f'), a.getAttribute('data-s'), a.getAttribute('data-e'));
    });
  }

  /* --------------------------------------------------------- index shell */

  function initIndex() {
    var side = document.getElementById('side');
    var frame = document.querySelector('#main iframe');
    if (!side || !frame) return;

    var links = side.querySelectorAll('a.nav');

    side.addEventListener('click', function (ev) {
      var a = ev.target.closest ? ev.target.closest('a.nav') : null;
      if (!a) return;
      for (var i = 0; i < links.length; i++) links[i].classList.remove('active');
      a.classList.add('active');
    });

    var box = document.getElementById('navsearch');
    if (box) {
      box.addEventListener('input', function () {
        var q = box.value.trim().toLowerCase();
        var groups = side.querySelectorAll('h2, .grp');
        for (var i = 0; i < links.length; i++) {
          var hit = !q || links[i].textContent.toLowerCase().indexOf(q) >= 0;
          links[i].style.display = hit ? '' : 'none';
        }
        for (var j = 0; j < groups.length; j++) groups[j].style.display = q ? 'none' : '';
      });
    }

    var tbtn = document.getElementById('themebtn');
    if (tbtn) {
      var sync = function () {
        var t = document.documentElement.getAttribute('data-theme');
        tbtn.textContent = t === 'light' ? '🌙' : '☀';
      };
      sync();
      tbtn.addEventListener('click', function () {
        var next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
        applyTheme(next); saveTheme(next); sync();
        try { frame.contentWindow.postMessage({ cw: 'theme', theme: next }, '*'); } catch (e) {}
      });
      frame.addEventListener('load', function () {
        var t = document.documentElement.getAttribute('data-theme');
        try { frame.contentWindow.postMessage({ cw: 'theme', theme: t }, '*'); } catch (e) {}
      });
    }
  }

  /* ------------------------------------------------- standalone source.html */

  function qs(name) {
    var m = new RegExp('[?&]' + name + '=([^&]*)').exec(location.search);
    return m ? decodeURIComponent(m[1].replace(/\+/g, ' ')) : '';
  }

  function initSourcePage() {
    var app = document.getElementById('svapp');
    if (!app) return;

    var list = document.getElementById('svlist');
    var filter = document.getElementById('svfilter');
    var hint = document.getElementById('svhint');
    var main = document.getElementById('svmain');

    var els = {
      path: main.querySelector('.sv-path'),
      badge: main.querySelector('.sv-badge'),
      note: main.querySelector('.sv-note'),
      body: main.querySelector('.sv-body'),
      btnPart: main.querySelector('[data-r="part"]'),
      btnFull: main.querySelector('[data-r="full"]')
    };
    var sf = new CWSrc.Surface(els);

    main.querySelector('.sv-tools').addEventListener('click', function (ev) {
      var b = ev.target.closest('[data-r]');
      if (!b) return;
      var r = b.getAttribute('data-r');
      if (r === 'part' || r === 'full') sf.setMode(r);
      else if (r === 'copy') copyText(sf.currentText(), b);
    });

    var tbtn = document.getElementById('themebtn');
    if (tbtn) {
      var sync = function () {
        tbtn.textContent = document.documentElement.getAttribute('data-theme') === 'light' ? '🌙' : '☀';
      };
      sync();
      tbtn.addEventListener('click', function () {
        var next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
        applyTheme(next); saveTheme(next); sync();
      });
    }

    var files = [];

    function paintList() {
      var q = (filter.value || '').trim().toLowerCase();
      var cur = sf.state.path;
      var out = [];
      var n = 0;
      for (var i = 0; i < files.length && n < 1200; i++) {
        if (q && files[i].toLowerCase().indexOf(q) < 0) continue;
        out.push('<a class="fitem' + (files[i] === cur ? ' active' : '') +
          '" href="#" data-p="' + CWSrc.esc(files[i]) + '" title="' + CWSrc.esc(files[i]) + '">' +
          CWSrc.esc(files[i]) + '</a>');
        n++;
      }
      list.innerHTML = out.join('') || '<div class="hint">无匹配文件</div>';
    }

    list.addEventListener('click', function (ev) {
      var a = ev.target.closest('[data-p]');
      if (!a) return;
      ev.preventDefault();
      sf.open(a.getAttribute('data-p'), null, null, 'full');
      paintList();
    });
    filter.addEventListener('input', paintList);

    function useIndex() {
      files = Object.keys(window.__WIKI_SRC_INDEX__ || {}).sort();
      hint.innerHTML = files.length
        ? ('静态模式 · ' + files.length + ' 个被引用文件<br>' +
           '用 <code>render.py serve</code> 可浏览整个仓库')
        : '暂无内嵌源码';
      paintList();
    }

    function boot() {
      if (CWSrc.isLive()) {
        fetch(CWSrc.ROOT + '__tree__')
          .then(function (r) {
            if (!r.ok) throw new Error('no server');
            var ct = r.headers.get('content-type') || '';
            if (ct.indexOf('json') < 0) throw new Error('no server');
            return r.json();
          })
          .then(function (j) {
            files = j.files || [];
            hint.textContent = '实时模式 · ' + files.length + ' 个文件' +
              (j.truncated ? '（已截断）' : '');
            paintList();
          })
          .catch(useIndex);
      } else { useIndex(); }
    }
    boot();

    var f = qs('f');
    if (f) sf.open(f, qs('s'), qs('e'), qs('mode') || (qs('s') ? 'part' : 'full'));
    else els.body.innerHTML = '<div class="sv-empty">从左侧选择一个文件开始浏览。</div>';
  }

  /* ---------------------------------------------------------------- boot */

  function boot() {
    initMermaid();
    initCodeBlocks();
    initSourceRefs();
    initIndex();
    initSourcePage();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else { boot(); }
})();
