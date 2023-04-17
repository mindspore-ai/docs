!function(n){var e={};function t(i){if(e[i])return e[i].exports;var o=e[i]={i:i,l:!1,exports:{}};return n[i].call(o.exports,o,o.exports,t),o.l=!0,o.exports;}t.m=n,t.c=e,t.d=function(n,e,i){t.o(n,e)||Object.defineProperty(n,e,{enumerable:!0,get:i});},t.r=function(n){'undefined'!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(n,Symbol.toStringTag,{value:'Module'}),Object.defineProperty(n,'__esModule',{value:!0});},t.t=function(n,e){if(1&e&&(n=t(n)),8&e)return n;if(4&e&&'object'==typeof n&&n&&n.__esModule)return n;var i=Object.create(null);if(t.r(i),Object.defineProperty(i,'default',{enumerable:!0,value:n}),2&e&&'string'!=typeof n)for(var o in n)t.d(i,o,function(e){return n[e];}.bind(null,o));return i;},t.n=function(n){var e=n&&n.__esModule?function(){return n.default;}:function(){return n;};return t.d(e,'a',e),e;},t.o=function(n,e){return Object.prototype.hasOwnProperty.call(n,e);},t.p='',t(t.s=0);}([function(n,e,t){t(1),n.exports=t(3);},function(n,e,t){(function(){var e='undefined'!=typeof window?window.jQuery:t(2);n.exports.ThemeNav={navBar:null,win:null,winScroll:!1,winResize:!1,linkScroll:!1,winPosition:0,winHeight:null,docHeight:null,isRunning:!1,enable:function(n){var t=this;void 0===n&&(n=!0),t.isRunning||(t.isRunning=!0,e((function(e){t.init(e),t.reset(),t.win.on('hashchange',t.reset),n&&t.win.on('scroll',(function(){t.linkScroll||t.winScroll||(t.winScroll=!0,requestAnimationFrame((function(){t.onScroll();})));})),t.win.on('resize',(function(){t.winResize||(t.winResize=!0,requestAnimationFrame((function(){t.onResize();})));})),t.onResize();})));},enableSticky:function(){this.enable(!0);},init:function(n){n(document);var e=this;this.navBar=n('div.wy-side-scroll:first'),this.win=n(window),n(document).on('click','[data-toggle=\'wy-nav-top\']',(function(){n('[data-toggle=\'wy-nav-shift\']').toggleClass('shift'),n('[data-toggle=\'rst-versions\']').toggleClass('shift');})).on('click','.wy-menu-vertical .current ul li a',(function(){var t=n(this);n('[data-toggle=\'wy-nav-shift\']').removeClass('shift'),n('[data-toggle=\'rst-versions\']').toggleClass('shift'),e.toggleCurrent(t),e.hashChange();})).on('click','[data-toggle=\'rst-current-version\']',(function(){n('[data-toggle=\'rst-versions\']').toggleClass('shift-up');})),n('table.docutils:not(.field-list,.footnote,.citation)').wrap('<div class=\'wy-table-responsive\'></div>'),n('table.docutils.footnote').wrap('<div class=\'wy-table-responsive footnote\'></div>'),n('table.docutils.citation').wrap('<div class=\'wy-table-responsive citation\'></div>'),n('.wy-menu-vertical ul').not('.simple').siblings('a').each((function(){var t=n(this);expand=n('<span class="toctree-expand"></span>'),expand.on('click',(function(n){return e.toggleCurrent(t),n.stopPropagation(),!1;})),t.prepend(expand);}));},reset:function(){var n=encodeURI(window.location.hash)||'#';try{var e=$('.wy-menu-vertical'),t=e.find('[href="'+n+'"]');if(0===t.length){var i=$('.document [id="'+n.substring(1)+'"]').closest('div.section');0===(t=e.find('[href="#'+i.attr('id')+'"]')).length&&(t=e.find('[href="#"]'));}t.length>0&&($('.wy-menu-vertical .current').removeClass('current'),t.addClass('current'),t.closest('li.toctree-l1').addClass('current'),t.closest('li.toctree-l1').parent().addClass('current'),t.closest('li.toctree-l1').addClass('current'),t.closest('li.toctree-l2').addClass('current'),t.closest('li.toctree-l3').addClass('current'),t.closest('li.toctree-l4').addClass('current'),t.closest('li.toctree-l5').addClass('current'),t[0].scrollIntoView());}catch(n){console.log('Error expanding nav for anchor',n);}},onScroll:function(){this.winScroll=!1;var n=this.win.scrollTop(),e=n+this.winHeight,t=this.navBar.scrollTop()+(n-this.winPosition);n<0||e>this.docHeight||(this.navBar.scrollTop(t),this.winPosition=n);},onResize:function(){this.winResize=!1,this.winHeight=this.win.height(),this.docHeight=$(document).height();},hashChange:function(){this.linkScroll=!0,this.win.one('hashchange',(function(){this.linkScroll=!1;}));},toggleCurrent:function(n){var e=n.closest('li');e.siblings('li.current').removeClass('current'),e.siblings().find('li.current').removeClass('current'),e.find('> ul li.current').removeClass('current'),e.toggleClass('current');}},'undefined'!=typeof window&&(window.SphinxRtdTheme={Navigation:n.exports.ThemeNav,StickyNav:n.exports.ThemeNav}),function(){for(var n=0,e=['ms','moz','webkit','o'],t=0;t<e.length&&!window.requestAnimationFrame;++t)window.requestAnimationFrame=window[e[t]+'RequestAnimationFrame'],window.cancelAnimationFrame=window[e[t]+'CancelAnimationFrame']||window[e[t]+'CancelRequestAnimationFrame'];window.requestAnimationFrame||(window.requestAnimationFrame=function(e,t){var i=(new Date).getTime(),o=Math.max(0,16-(i-n)),r=window.setTimeout((function(){e(i+o);}),o);return n=i+o,r;}),window.cancelAnimationFrame||(window.cancelAnimationFrame=function(n){clearTimeout(n);});}();}).call(window);},function(n,e){n.exports=jQuery;},function(n,e,t){}]);



// 公共css/js文件
(function () {
  var s = document.getElementsByTagName('HEAD')[0];
  let origin = window.location.origin;
  var hm = document.createElement('script');
      hm.src = origin + '/common.js';
  var oLink = document.createElement('link');
  oLink.rel = 'stylesheet';
  oLink.href = origin + '/h5_docs.css';
  s.parentNode.insertBefore(hm, s);
  s.parentNode.insertBefore(oLink, s);
})();

// 模板2
$(function () {
  $('body').addClass('theme-tutorials');

  const pathname = window.location.pathname;
  const isEn = pathname.indexOf('/en/') !== -1;
  const lang = isEn?'/en/':'/zh-CN/';
  const pathPrefix = pathname.split(lang);
  const currentVersion = pathPrefix[1].split('/')[0];
  const pagePath = pathPrefix[0] +lang + currentVersion;

  let msDocsVersion = [],
      versionDropdownList = [],
      msVersionInfo = [],
      pageTitle ='';
  
  // 获取当前版本 不带R
  function curVersion(version) {
      return version === 'master'
          ? 'master'
          : version.startsWith('r') ? version.slice(1):version;
  }

  // 请求数据
  function getHeaderData(url) {
      return new Promise((resolve, reject) => {
          $.ajax({
              type: 'get',
              url: url,
              dataType: 'json',
              success: (res) => {
                  resolve(res);
              },
              error: (e) => {
                  reject(e);
              }
          });
      });
  }

  // 版本名称显示
  function pageVersionName (){
    let name= '';
    versionDropdownList.forEach((subitem) => {
      if(currentVersion.endsWith(subitem.version)){
        name= subitem.versionAlias !==''?subitem.versionAlias:subitem.version;
      }
    });
    if (versionName === '') {
      versionName = currentVersion;
    }
    return curVersion(name);
  }


  // 切换版本下拉菜单
  function versionDropdown(obj) {
      return `<div class='version-select-wrap'><div class="version-select-dom">
        <span class="versionText">${pageVersionName()}</span> <img src="/pic/down.svg" />
            <ul>
                ${obj.map(function (item) {
                  if(item.active){
                    return item.versions.map((subitem) => {
                      return `<li><a href="${subitem.url}" class='version-option'>${subitem.versionAlias !==''?curVersion(subitem.versionAlias):subitem.version}</a></li>`;
                    }).join('');
                  }
                }).join('')}
              <li><a href="/versions/${ isEn ? 'en' : ''}" class='version-option'>${isEn?'More':'更多'}</a></li>
            </ul>
        </div></div>`;
  }

  const initPage = async function () {
      msDocsVersion = await getHeaderData('/msVersion.json');
      msVersionInfo = await getHeaderData(`${pagePath}/_static/js/version.json`);

      pageTitle = isEn ? msVersionInfo.label.en || '': msVersionInfo.label.zh || '';
      const pageSubMenu = isEn ? msVersionInfo.submenu.en  || []: msVersionInfo.submenu.zh  || [];

      let theme2Nav = '';
      msDocsVersion.forEach(function (item) { 
              versionDropdownList = item.versions.slice(0,3);
              // 格式化版本拉下菜单
              pageSubMenu.forEach((item) => {
                  item.versions = versionDropdownList.map((sub) => {
                    if (item.url.includes(pagePath)) {
                      item.active = 1;
                    }
                    return{
                        version: curVersion(sub.version),
                        url: sub.url !=='' ? sub.url:item.url.replace(currentVersion, sub.version),
                        versionAlias: sub.versionAlias
                    };
                });
              });

              // 生成菜单
              theme2Nav = `<nav class="header-wapper row navbar navbar-expand-lg navbar-light header-wapper-lite header-wapper-docs" >
              <div class="header-nav navbar-nav" style="height:100%;justify-content: flex-end;"><div class="bottom" style="line-height: initial;">
              ${pageSubMenu
                  .map(function (item) {
                      return `<div class="header-nav-link">
                          <a class="header-nav-link-line ${
                              item.active ? 'selected' : ''
                          }" href="${item.url}">${item.label}</a>
                      </div>
                      `;
                  })
                  .join('')}
        </div></nav>`;
           
      });

      // 教程首页中间导航点击在本页打开
    $('.toctree-l1 .reference').on('click', function (e) {
        e.preventDefault();
        window.open($(this).attr('href'), ($(this).attr('target') || '_self'));
    });

      setTimeout(() => {
          $('.header').append(theme2Nav);
          $('.wy-breadcrumbs>li:first-of-type')[0].innerText =
              pageTitle + ' (' + pageVersionName() + ')';
          $('#rtd-search-form input').attr(
              'placeholder',
              isEn ? 'Search in Tutorials' : '"教程" 内搜索'
          );
          let welcomeText = isEn ? 'MindSpore Tutorials': '欢迎查看MindSpore教程';
          $('.wy-menu-vertical').before(
            `<div class="docsHome"><a  href="#" class="welcome">${welcomeText}</a></div>`
          );

          // 版本选择
          let width = window.innerWidth;
          if (width < 768) {
              $('#nav-h5').append(versionDropdown(pageSubMenu));
              $('#nav-h5 .version-select-dom').on('click', function () {
                  $(this).find('ul').slideToggle();
              });
          } else {
              $('.wy-nav-side')
                  .addClass('side-fix')
                  .prepend(versionDropdown(pageSubMenu));
          }
          if (pathname.indexOf('/index.html') !== -1) {
            $('.welcome').addClass('selected');
          }
      }, 100);

      // 左侧菜单控制
      let aList = $('.wy-menu-vertical>ul>.current>ul>.toctree-l2>a');
      for (let i = 0; i < aList.length; i++) {
          let hash = aList[i].hash;
          if (hash != '') {
              aList[i].parentNode.parentNode.style.display = 'none';
              aList[i].parentNode.parentNode.parentNode.className = aList[i].parentNode.parentNode.parentNode.className + ' ' + 'navNoPlus';
          }
      }
      
  };

  initPage();
});
