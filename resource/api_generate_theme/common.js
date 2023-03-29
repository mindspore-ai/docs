// 埋点JS
function createScriptSensor() {
    var oHead = document.getElementsByTagName('HEAD').item(0);
    let sensors_origin = location.origin;
    let jsScript1 = document.createElement('script');
    let jsScript2 = document.createElement('script');
    let jsScript3 = document.createElement('script');
    jsScript1.src = `${
        location.protocol === 'https:' ? 'https:' : 'http:'
    }//pv.sohu.com/cityjson?ie=utf-8`;
    jsScript2.src = sensors_origin + '/allow_sensor/sensorsdata.min.js';
    jsScript3.src = sensors_origin + '/allow_sensor/sensors.js';
    oHead.appendChild(jsScript1);
    oHead.appendChild(jsScript2);
    setTimeout(() => {
        oHead.appendChild(jsScript3);
    });
}

function createScriptCommonJs() {
    let oHead = document.getElementsByTagName('HEAD').item(0);
    let origin = location.origin;
    let jsScript = document.createElement('script');
    jsScript.type = 'text/javascript';
    jsScript.src = origin + '/commonJs/docHandle.js';
    setTimeout(() => {
        oHead.appendChild(jsScript);
    });
}

// 百度统计
function createScriptBaidu() {
    // var _hmt = _hmt || [];
    (function () {
        var hm = document.createElement('script');
        hm.src = 'https://hm.baidu.com/hm.js?7c2afdec4c0d635d30ebb361804d0464';
        var s = document.getElementsByTagName('script')[0];
        s.parentNode.insertBefore(hm, s);
    })();
}

// jQuery中id含有特殊字符转义后使用
function escapeJquery(srcString) {
  // 转义之后的结果
  var escapseResult = srcString;
  // javascript正则表达式中的特殊字符
  var jsSpecialChars = ['\\', '^', '$', '*', '?', '.', '+', '(', ')', '[', ']', '|', '{', '}'];
  // jquery中的特殊字符,不是正则表达式中的特殊字符
  var jquerySpecialChars = ['~', '`', '@', '#', '%', '&', '=', '\'', '"',	':', ';', '<', '>', ',', '/'];
  for (let i = 0; i < jsSpecialChars.length; i++) {
    escapseResult = escapseResult.replace(
      new RegExp('\\'+ jsSpecialChars[i], 'g'),
      '\\' + jsSpecialChars[i]
    );
  }
  for (let i = 0; i < jquerySpecialChars.length; i++) {
    escapseResult = escapseResult.replace(
      new RegExp(jquerySpecialChars[i],'g'),
      '\\' + jquerySpecialChars[i]
    );
  }
  return escapseResult;
}

// 判断是否 h5
function isH5(cb) {
  let screen = document.documentElement.clientWidth;
  if (screen < 768) {
      cb();
  }
}

$(function () {
  // 统一修改title
  $('title').text('MindSpore');
  const pathname = window.location.pathname;
  const isEn = pathname.indexOf('/en/') !== -1;
  const lang = isEn?'/en/':'/zh-CN/';
  const enPath =isEn?'en':'';

  //切换语言链接
  const newNavPath = isEn ? pathname.replace('/en/','/zh-CN/'):pathname.replace('/zh-CN/','/en/');

  // 获取当前版本
  const getCurrentVersion = () => {
    let currentVersion = 'master';
    if(pathname.startsWith('/docs/') || pathname.startsWith('/tutorials/zh-CN/') || pathname.startsWith('/tutorials/en/')){
      currentVersion = pathname.split('/')[3];
    }else{
      currentVersion = pathname.split('/')[4];
    }
    return pathname.includes('/master/')?'master': currentVersion.startsWith('r')&& currentVersion.slice(1);
  };

  // 文档教程搜索埋点记录
  const sensorsMethods = {
    getSearchKey: function () {
        var params = $.getQueryParameters();
        if (params.q) {
            sensorsMethods.startSensor(params.q[0], 5);
        }
    },
    startSensor: function (search_key, num) {
        if (!num) {
            return;
        }
        if (window['sensorsCustomBuriedData']) {
            sensorsMethods.searchBuriedData(search_key);
        } else {
            num--;
            setTimeout(() => {
                // 若是一开始没有值，则重试
                sensorsMethods.startSensor(search_key, num);
            }, 2000);
        }
    },
    searchBuriedData: function (search_key) {
        if (window['sensorsCustomBuriedData']) {
            const search_event_id = `${search_key}${new Date().getTime()}${
                window['sensorsCustomBuriedData'].ip || ''
            }`;
            const obj = {
                search_key,
                search_event_id
            };
            window['addSearchBuriedData'] = obj;
            let sensors = window['sensorsDataAnalytic201505'];
            if (sensors) {
                sensors.setProfile({
                    profileType: 'searchValue',
                    ...(window['sensorsCustomBuriedData'] || {}),
                    ...(window['addSearchBuriedData'] || {})
                });
                sensorsMethods.selectBuriedData();
            }
        }
    },
    // 选中文档埋点
    selectBuriedData: function () {
        const data = $('#search-results > .search > li > a');
        if (data.length) {
            data.each((index, item) => {
                $(item).on('click', function (e) {
                    let sensors = window['sensorsDataAnalytic201505'];
                    let search_tag = '';
                    if (location.pathname.includes('tutorial')) {
                        search_tag = 'tutorial';
                    } else if (location.pathname.includes('docs')) {
                        search_tag = 'docs';
                    }
                    const searchKeyObj = {
                        search_tag,
                        search_rank_num: index + 1,
                        search_result_total_num: data.length,
                        search_result_url: e.currentTarget.href
                    };
                    sensors.setProfile({
                        profileType: 'selectSearchResult',
                        ...(window['sensorsCustomBuriedData'] || {}),
                        ...(window['addSearchBuriedData'] || {}),
                        ...searchKeyObj
                    });
                });
            });
        }
    }
  };

  const body = $('body');

  let headerMenuData = [],
  msDocsVersion = [];
  // 获取导航菜单
  function getHeaderData(url){
    return new Promise((resolve,reject) => {
        $.ajax({
          type: 'get',
          url: url,
          dataType: 'json',
          success: (res) => {
            resolve(res);
          },
          error:(e) => {
            reject(e);
          }
        });
    });
  }

  //初始化header
  const msHeader = {
    pcHeader: function(){
      return `<header class="header">
        <nav class="header-wapper">
            <div class="header-nav " style="display: flex;">
                <a href="/${enPath}" class="logo">
                    <img src="/pic/${isEn ?  'logo_black_en.png' : 'logo_black.png'}" alt="logo" />
                </a>
                ${headerMenuData&&headerMenuData.map(function (item) {
                    if(!item.children) {
                        return `
                        <div class="header-nav-link"><a class="header-nav-link-line" href="${item.url ===''? msHeader.headerNavLinks(item.id):item.url}">${item.name}</a></div>
                        `;
                    } else {
                        if (pathname.startsWith('/' + item.id)) {
                          item.active = 1;
                        }
                        return `
                        <div class="header-nav-link">
                          <a class="header-nav-link-line ${item.active ? 'selected' : ''}" href="${item.url ==='' || item.url === undefined ? msHeader.headerNavLinks(item.id):item.url}">${item.name}</a>
                            <div class="dropdown-menu-git ${item.id==='docs'?'dropdown-menu-docs':''}" >
                                    ${item.children.map(function (sub) {
                                        if(sub.children){
                                          return `<div class="docsNew-column">
                                                    <div class="docsNew-column-title">${sub.title || ''}</div>
                                                <div class="bottom" style="line-height: initial;">
                                                    ${sub.children.map(function (sub) {
                                                        return `
                                                        <div class="docsVersion"><a class="versionM" href="${msHeader.headerNavLinks(sub.id)}">${sub.name}</a></div>
                                                        `;
                                                    }).join('')}
                                                </div>
                                            </div>`;
                                        }else{
                                          return `<li><a target="${sub.openType === undefined?'_self':sub.openType}" href="${sub.url==='' || sub.url === undefined? msHeader.headerNavLinks(sub.id):sub.url }">${sub.name}</a></li>`;
                                        }
                                    }).join('')}
                                </div>
                        </div>`;
                    }

                }).join('')}
            </div>
            <div class="header-nav navbar-nav" style="display: flex;">
                <div class="dropdown">
                    <a href="${newNavPath}" class="dropdown-toggle" id="dropdownMenu2" data-toggle="dropdown" aria-haspopup="true" aria-expanded="true">
                        <span class="languageIpt">EN</span>
                    </a>
                </div>
                <div class="header-nav-link">
                    <p style="cursor: pointer;">${isEn?'Code':'代码'}</p>
                    <ul class="dropdown-menu-git">
                        <li><a href="https://gitee.com/mindspore/mindspore" target="_blank">Gitee</a></li>
                        <li><a href="https://github.com/mindspore-ai/mindspore" target="_blank">GitHub</a></li>
                    </ul>
                </div><a class="header__search"></a>
                <div class="searchMain" style="display: none;">
                    <div class="searchInput"><span class="search-icon"></span><span class="close-icon"></span><input
                            class="search-val" placeholder="${isEn?'Site-wide search':'全站搜索'}"></div>
                    <ul class="hotWord"></ul>
                </div>
            </div>
        </nav>
    </header>
      `;
    },
    h5Header: function(){
      return `<header id="header-h5" class="header-mobile">
      <div class="ms-content">
          <div class="header--left">
              <a href="/${enPath}"><img src="/pic/${isEn ? 'logo_black_en.png' : 'logo_black.png'}" alt="logo" /></a>
          </div>
          <div class="header--right">
              <div class="pc-lang-switch">
                  <div class="pc-lang-current">
                      <a href="/search/${enPath}" class="search"></a>
                      <a href="${newNavPath}" class="dropdown-toggle" id="dropdownMenu1" data-toggle="dropdown" aria-haspopup="true" aria-expanded="true">
                      <span class="languageIpt">EN</span>
                  </a>
                  </div>
              </div>
              <div class="more-nav" show="1"></div>
          </div>
          <div class="nav-link-container">
              ${headerMenuData&&headerMenuData.map(function (item) {
                      if (!item.children) {
                          return `<div class="mobile-nav-item"><a class="mobile-nav-link" href="${item.url === undefined ?item.url: msHeader.headerNavLinks(item.id) }">${item.name} </a></div>`;
                      } else {
                          return `<div class="mobile-nav-item drop-nav">
                                    <div class="mobile-nav-link">${item.name}<span class="btnArrow topArrow"></span></div>
                                    <div class="mobile-subnav-wraper">
                                          ${item.children.map(function (item) {
                                            if(item.children){
                                              return `${item.children.map(function (sub) {
                                                return `<a class="mobile-subnav-link" href="${msHeader.headerNavLinks(sub.id)}">${sub.name}</a>`;
                                            }).join('')}`;
                                            }else{
                                              return `<a target="${item.openType===undefined?'_self':item.openType}" class="mobile-subnav-link" href="${item.url ==='' || item.url === undefined? msHeader.headerNavLinks(item.id):item.url}">${item.name}</a>`;
                                            }
                                          }).join('')}
                                    </div>
                                  </div> `;
                      }
                  }).join('')}
          </div>
      </div>
  </header>`;
    },
    h5Nav: function(){
      return `
      <div class="nav-h5" id="nav-h5" style="height:4rem;"><div class="nav" show="1"></div></div>`;
    },
    // 获取页面标题
    pageTitle:function(){
      let title = '';
      msDocsVersion&&msDocsVersion.forEach((item) => {
        if(pathname.startsWith('/'+item.name)){
          title = isEn?item.label.en:item.label.zh;
        }
      });
      return title;
    },
    // 文档页面交互
    headerMethods:function (){
        const searchUrl = isEn?'/search/en':'/search';
            // 点击切换语言开始
        if (isEn) {
            $('.languageIpt').text('中');
            $('.changeTutorial').css('left', 0);
            $('.logo').attr('href', '/en');
        } else {
            $('.languageIpt').text('EN');
        }
        // 点击搜索弹出搜索框
        $('.header__search').on('click', function () {
            $(this).css('display', 'none').prevAll().css('display', 'none');
            $('.dropdown').css('display', 'none !important');
            $('.searchMain').css('display', 'block');
        });
        // 点击关闭搜索框
        $('.close-icon').on('click', function () {
            $('.header__search')
            .css('display', 'block')
            .prevAll()
            .css('display', 'block');
            $('.dropdown').css('display', 'flex !important');
            $('.searchMain').css('display', 'none');
        });
        // 获取搜索框的值并传递到搜索页面
        $('.search-val').on('keydown', function (e) {
            const val = $('.search-val').val();
            if (e.keyCode === 13 && val !== '') {
            window.location.href = searchUrl + '?inputValue=' + val;
            }
        });
        $('.search-icon').on('click', function () {
            const val = $('.search-val').val();
            if (val !== '') {
            window.location.href = searchUrl + '?inputValue=' + val;
            }
        });
        // 搜索框联想词设置
        $('.search-val').on('input', function () {
            let val = $('.search-val').val();
            let $hotWord = $('.hotWord');
            $.ajax({
            type: 'get',
            url: '/tips?keywords=' + val + '&index=mindspore_index_tips',
            dataType: 'json',
            success: function (res) {
                if (res && res.status && res.status === 200) {
                let arr = res.obj ? res.obj : [];
                $hotWord.html('');
                let html = '';
                if (arr.length > 0) {
                    arr.map(function (item, index) {
                    html +=
                        '<li class="search--list" key=' +
                        index +
                        '>' +
                        item +
                        '</li>';
                    });
                    $hotWord.append(html);
                }
                $('.search--list').on('click', function (e) {
                    let value = e.target.innerText;
                    window.location.href = '/search?inputValue=' + value;
                    $('.header-nav').css('display', 'flex');
                    $('.searchMain').css('display', 'none');
                });
                }
            }
            });
        });
        // 点击页面其余地方搜索框消失
        $(document).mousedown(function (e) {
            const target = $(e.target)[0].className;
            if (
            target === 'header__search' ||
            target === 'search-val' ||
            target === 'search-icon' ||
            target === 'search--list'
            ) {
            $('.header__search')
                .css('display', 'none')
                .prevAll()
                .css('display', 'none');
            $('.header-nav').css('display', 'flex');
            $('.searchMain').css('display', 'block');
            } else {
            $('.header__search')
                .css('display', 'block')
                .prevAll()
                .css('display', 'block');
            $('.searchMain').css('display', 'none');
            }
        });
    },
    h5HeaderMethods:function(){
      msFotter.documentEvaluationFn();
      // 国际化
      $('#dropdownMenu_h5').on('click', function () {
        $('.dropdown-menu').slideToggle();
      });
      // 主导航显示
      $('.more-nav').on('click', function () {
          if ($(this).attr('show') == '0') {
            msHeader.hideNav(0);
              $(this).css('backgroundImage', 'url("/pic/nav_white.png")');
          } else {
            msHeader.showNav(0);
              $(this).css('backgroundImage', 'url("/pic/close.png")');
          }
      });

      // 侧栏导航显示
      $('.nav').on('click', function () {
          if ($(this).attr('show') == '0') {
            msHeader.hideNav(1);
          } else {
            msHeader.showNav(1);
          }
      });

      // 主导航二级菜单
      $('.drop-nav').each(function () {
          $(this).click(function () {
              $('.mobile-subnav-wraper').slideUp();
              $('.btnArrow').css('transform', 'rotate(0deg)');
              if ($(this).children('.mobile-subnav-wraper')[0].style.display === 'block') {
                  $(this).children('.mobile-subnav-wraper').slideUp();
                  $(this).children('.mobile-nav-link').children('.btnArrow').css('transform', 'rotate(0deg)');

              } else {
                  $(this).children('.mobile-subnav-wraper').slideDown();
                  $(this).children('.mobile-nav-link').children('.btnArrow').css('transform', 'rotate(180deg)');
              }
          });
      });

      // mask点击关闭侧栏
      $('#mask').on('click', function () {
        msHeader.hideNav(0);
        msHeader.hideNav(1);
      });
    },
    headerPathVersion:function(path) {
      let version = 'master';
       msDocsVersion&&msDocsVersion.forEach((item) => {
        if(path.includes(item.name)){
          if(typeof item.versions[1] === 'string'){
            version = item.versions.length >1 ? item.versions[1] : item.versions[0];
          }else{
            version =item.versions[1].version;
          }
        }
      });
      return version;
    },
    headerNavLinks:function(path){
      let href = '';
      if(path ==='lite'){
        href = '/'+path;
      }else if(path==='tutorials'|| path ==='docs'){
        href = `/${path + lang + msHeader.headerPathVersion(path)}/index.html`;
      }else{
        href = `/${path+'/docs' + lang + msHeader.headerPathVersion(path)}/index.html`;
      }
      return href;
    },
    // 导航显示 隐藏  val : 0 主导航  val : 1 侧导航
    showNav:function(val) {
        if (val) {
            $('.wy-nav-side').css({ 'left': '0', 'transition': '0.3s' });
            $('#side-nav').css({ 'left': '0', 'transition': '0.3s' });
            $('.nav').attr('show', '0');
            $('#mask').css('zIndex', '10');
        } else {
            $('.nav-link-container').css('right', 0);
            $('.more-nav').attr('show', '0');
            $('#mask').css('zIndex', '150');
            this.hideNav(1);
        }
        body.addClass('overflow');
        $('#mask').show();
    },
    hideNav:function(val)  {
        if (val) {
            $('.wy-nav-side').css({ 'left': '-90%', 'transition': '0.3s' });
            $('#side-nav').css({ 'left': '-90%', 'transition': '0.3s' });
            $('.nav').attr('show', '1');
        } else {
            $('.nav-link-container').css('right', '-65%');
            $('.mobile-subnav-wraper').slideUp();
            $('.btnArrow').css('transform', 'rotate(0deg)');
            $('.more-nav').attr('show', '1').css('backgroundImage', 'url("/pic/nav_white.png")');
        }
        body.removeClass('overflow');
        $('#mask').hide();
    }
  };

  // 初始化footer
  const msFotter = {
    fontmatter: {
      askQuestionHref:'',
      askQuestion3:isEn?'Document Feedback':'文档反馈',
      helpforyou:isEn?'Was this helpful?':'这个对你有帮助吗 ?',
      askQuestion2:isEn?'Quick Feedback':'快速反馈问题',
      askQuestionInfo:isEn?'Click here to commit an issue in the code repository. Describe the issue in the template. We will follow up on it.':'点击图标，可跳转代码仓提issue，按照issue模板填写问题描述，我们将会跟进处理',
      askQuestionInfo2:isEn?'Remember to add the tag below:':'记得添加mindspore-assistant标签哦！',
      helpfor1:isEn?'Not helpful':'基本没有用',
      helpfor2:isEn?'Somewhat helpful':'有一点帮助',
      helpfor3:isEn?'Helpful':'基本上能用',
      helpfor4:isEn?'Very helpful':'能解决问题',
      helpfor5:isEn?'Totally helpful':'文档很有用',
      install:isEn?'Install':isEn?'':'安装',
      tutorial:isEn?'Tutorials':'教程',
      docs:isEn?'Docs':'文档',
      community:isEn?'Community':'社区',
      news:isEn?'News':'资讯',
      security:isEn?'Security':'安全',
      forum:isEn?'Ascend Forum':'论坛',
      knowledgeMap:isEn?'Knowledge Map':'知识地图',
      forumText:isEn?'Ask questions in Ascend Forum':'到论坛去提问',
      forumPath:isEn?'https://forum.huawei.com/enterprise/en/forum-100504.html':'https://www.hiascend.com/forum/forum-0106101385921175002-1.html',
      copyRight:isEn?'Copyright©MindSpore 2022':'版权所有©MindSpore 2022',
      terms:isEn?'Terms of Use':'法律声明',
      privacy:isEn?'Privacy Policy':'个人信息保护政策',
      license:isEn?'The content of this page is licensed under the<a target=\'_blank\' href=\'https://creativecommons.org/licenses/by/4.0/\'> Creative Commons Attribution 4.0 License</a>, and code samples are licensed under the <a target=\'_blank\' href=\'https://www.apache.org/licenses/LICENSE-2.0\'>Apache 2.0 License</a>.':'本页面的内容根据<a target=\'_blank\' href=\'https://creativecommons.org/licenses/by/4.0/\'>Creative Commons Attribution 4.0</a>许可证获得许可，代码示例根据<a target=\'_blank\' href=\'https://www.apache.org/licenses/LICENSE-2.0\'>Apache 2.0</a>许可证获得许可。'
    },
    // 文档反馈链接
    getQuestionHref: function(){
      let url = '';
      if (pathname.startsWith('/docs/') && pathname.indexOf('/api_python/') >-1){
          url = 'https://gitee.com/mindspore/mindspore/issues/new?issue%5Bassignee_id%5D=0&issue%5Bmilestone_id%5D=0';
      }else {
          url = 'https://gitee.com/mindspore/docs/issues/new?issue%5Bassignee_id%5D=0&issue%5Bmilestone_id%5D=0';
      }
      msFotter.fontmatter.askQuestionHref = url;
    },
    // PC footer
    pcFootHTML: function (){
      return `<div id="footer">
          <div class="evaluate">
          <div class="evaluateTitle">
              ${msFotter.fontmatter.helpforyou}
              <div class="docsLayer">
                <div class="km-item">
                    <a class="km-item-link" href="/resources/knowledgeMap/${enPath}">
                      <img class="btn-img" src="/pic/knowledgeMap-icon.png"/>
                      <div class="btn-label">${msFotter.fontmatter.knowledgeMap}</div>
                    </a>
                </div>
                <div class="askQuestion-box">
                    <a class="askQuestion" href="${msFotter.fontmatter.askQuestionHref}" target="_blank">
                        <img class="btn-img" src="/pic/docs/ask.png"/>
                        <div class="btn-label">${msFotter.fontmatter.askQuestion3}</div>
                    </a>
                    <div class="askQuestion-info">
                        <div class="askQuestion-info-title"><img src="/pic/docs/text.png"/>${msFotter.fontmatter.askQuestion2}</div>
                        <div class="askQuestion-info-content">
                            <p class="first">${msFotter.fontmatter.askQuestionInfo}</p>
                            <p>${msFotter.fontmatter.askQuestionInfo2}</p>
                            <p><span></span> mindspore-assistant</p>
                        </div>
                    </div>
                </div>
              </div>
          </div>
          <ul class="evaluateStar">
              <li>
              <div class="star"></div>
              <div class="wordScore">${msFotter.fontmatter.helpfor1}</div>
              </li>
              <li>
              <div class="star"></div>
              <div class="wordScore">${msFotter.fontmatter.helpfor2}</div>
              </li>
              <li>
              <div class="star"></div>
              <div class="wordScore">${msFotter.fontmatter.helpfor3}</div>
              </li>
              <li>
              <div class="star"></div>
              <div class="wordScore">${msFotter.fontmatter.helpfor4}</div>
              </li>
              <li>
              <div class="star"></div>
              <div class="wordScore">${msFotter.fontmatter.helpfor5}</div>
              </li>
          </ul>
          </div>
          <div class="licensed">${msFotter.fontmatter.license}</div>
          <div class="partLine"></div>
          <div class="footer row">
          <a class="jump col-xs-12 col-md-4" target="_top" href="/install/${enPath}">${msFotter.fontmatter.install}</a>
          <a
              class="jump col-xs-12 col-md-4"
              target="_top"
              href="/tutorials/${isEn?'en':'zh-CN'}/master/index.html"
              >${msFotter.fontmatter.tutorial}</a
          ><a
              class="jump col-xs-12 col-md-4"
              target="_top"
              href="/mindspore/${enPath}"
              >${msFotter.fontmatter.docs}</a
          ><a
              class="jump col-xs-12 col-md-4"
              target="_top"
              href="/community/${enPath}"
              >${msFotter.fontmatter.community}</a
          ><a class="jump col-xs-12 col-md-4" target="_top" href="/news/${enPath}"
              >${msFotter.fontmatter.news}</a
          ><a class="jump col-xs-12 col-md-4" target="_top" href="/security/${enPath}"
              >${msFotter.fontmatter.security}</a
          ><a class="jump col-xs-12 col-md-4" target="_blank" href="${msFotter.fontmatter.forumPath}"
              >${msFotter.fontmatter.forum} </a
          >
          </div>
          <div class="row copyright col-xs-12 col-md-8">
          <span class="copyRight">${msFotter.fontmatter.copyRight}</span
          ><a class="copynum" target="_blank" href="https://beian.miit.gov.cn"
              >粤A2-20044005号</a
          ><a href="/legal/${enPath}" class="legal">${msFotter.fontmatter.terms}</a
          ><span class="verticalLine"></span
          ><a href="/privacy/${enPath}" class="privacyPolicy">${msFotter.fontmatter.privacy}</a>
          </div>
          <a
          class="footer-record"
          href="https://beian.miit.gov.cn"
          ><img class="copyImg2" src="/pic/copyright3.png" alt="img" /><span
              class="keepRecord"
              >粤公网安备 </span
          ><span class="recordNum">44030702002890号</span></a
          >
      </div>
      `;
    },
    //跳转论坛统计
    jumpForumStatistics: function(){
        $('.askQuestion').on('click', function() {
          $.ajax({
              type: 'POST',
              url: '/saveEssayJump',
              contentType: 'application/json',
              data: JSON.stringify({
                  essayUrl:location.href
              })
          });
      });
    },
    // H5 footer
    h5FootHTML:function(){
      return `
      <div id="h5_footer">
          <div class="evaluate">
          <div class="evaluateTitle">
              ${msFotter.fontmatter.helpforyou}
              <a class="askQuestion" style="padding-bottom:1px; border-bottom:1px solid #379bbe6;" href="${msFotter.fontmatter.forumPath}" target="_blank">${msFotter.fontmatter.forumText}</a>
          </div>
          <ul class="evaluateStar">
              <li>
              <div class="star"></div>
              <div class="wordScore">${msFotter.fontmatter.helpfor1}</div>
              </li>
              <li>
              <div class="star"></div>
              <div class="wordScore">${msFotter.fontmatter.helpfor2}</div>
              </li>
              <li>
              <div class="star"></div>
              <div class="wordScore">${msFotter.fontmatter.helpfor3}</div>
              </li>
              <li>
              <div class="star"></div>
              <div class="wordScore">${msFotter.fontmatter.helpfor4}</div>
              </li>
              <li>
              <div class="star"></div>
              <div class="wordScore">${msFotter.fontmatter.helpfor5}</div>
              </li>
          </ul>
          </div>
          <div class="licensed">${msFotter.fontmatter.license}</div>
          <div class="footer row">
          <div class="logo"><a href="/">
              <img src="/pic/${isEn ?  'logo_bottom_en.png' : 'logo_bottom.png'}" alt="logo" />
          </a></div>
          <div class="copyright">
              <div class="copynum">
              <p>
                  <span class="keepRecord">${msFotter.fontmatter.copyRight}</span>
                  <a target="_blank" href="https://beian.miit.gov.cn"
                  >粤A2-20044005号</a
                  >
              </p>
              <p>
                  <span class="keepRecord">粤公网安备 </span
                  ><a
                  class="footer-record"
                  href="https://beian.miit.gov.cn"
                  >44030702002890号</a
                  >
              </p>
              </div>
              <div class="legal">
              <a href="/legal/${enPath}" class="legal">${msFotter.fontmatter.terms}</a
              ><span class="verticalLine"></span
              ><a href="/privacy/${enPath}" class="privacyPolicy">${msFotter.fontmatter.privacy}</a>
              </div>
          </div>
          </div>
      </div>
      `;
    },

    srollEvent: function(){
      let startY, moveEndY, Y;
        let main = document.getElementsByTagName('body')[0];
        main.addEventListener('touchstart', function (e) {
            startY = e.touches[0].pageY;
        }, false);
        main.addEventListener('touchmove', function (e) {
            // 出现mask的时候，header禁止上移
            if ($('.nav').attr('show') == 1 && $('.more-nav').attr('show') == 1) {
                moveEndY = e.changedTouches[0].pageY;
                Y = moveEndY - startY;
                if (Y > 10) {
                    $('#header-h5').css('top', '0rem');
                    $('.nav-h5').css('top', '6rem');
                    $('.wy-breadcrumbs').css('top', '10rem');
                } else if (Y < -100) {
                    $('#header-h5').css('top', '-6rem');
                    $('.nav-h5').css('top', '-12rem');
                    $('.wy-breadcrumbs').css('top', '-12rem');
                }
            }
        });
    },
    documentEvaluationFn(){
      const star = $('div.star');
      star.on('mouseover', function () {
          $(this).addClass('sel');
          $(this).parent('li').prevAll().children('.star').addClass('sel');
          $(this).parent('li').nextAll().children('.star').removeClass('sel');
      }).on('mouseout', function () {
          $(this).removeClass('sel');
          $(this).parent('li').prevAll().children('.star').removeClass('sel');
      });
      $('.evaluateStar').on('click', '.star', function() {
          $(this).addClass('sel');
          $(this).parent('li').prevAll().children('.star').addClass('sel');
          $(this).parent('li').nextAll().children('.star').removeClass('sel');
          star.unbind('mouseover');
          star.unbind('mouseout');
          let grade = $(this).parent('li').prevAll().length + 1;
          $.ajax({
              type: 'POST',
              contentType: 'application/json',
              data: JSON.stringify({
                  score: grade,
                  essayUrl:location.href
              }),
              url:'/saveEssayScore'
          });
      });
    }
  };

  function isH5Show(){
    $('.wy-nav-content').append(msFotter.h5FootHTML).append('<div id="mask"></div>');
    $('#footer').remove();
    msFotter.srollEvent();
    body.prepend(msHeader.h5Header);
    $('.wy-nav-top').append(msHeader.h5Nav);
    setTimeout(() => {
      $('header.header').css({'display':'none'});
      msHeader.h5HeaderMethods();
    },50);
  }

  // 用一个宽度来记录之前的宽度，当之前的宽度和现在的宽度，从手机切换到电脑或者从电脑切换到手机，才执行下面部分函数
  const watchWinResize = () => {
    let oldWidth=window.innerWidth;
    window.addEventListener('resize', function() {
        let width=window.innerWidth;
        let h5Head=document.getElementById('header-h5');
        let h5footer=document.getElementById('h5_footer');
        let navh5=document.getElementById('nav-h5');
        let pcfooter=document.getElementById('footer');
        if(width<768){
            $('#mask').css('display', 'none');
            $('.wy-nav-side').css({ 'left': '-90%', 'transition': '0s' });
            if(h5Head===null){
              isH5Show();
            }
        }else{
            $('.header').css('display', 'block');
            $('#mask').css('display', 'none');
            $('.wy-nav-side').css('left', '0');
            if(h5Head || h5footer || navh5){
              h5Head.remove();
              h5footer.remove();
              navh5.remove();
            }
            if(pcfooter===null){
              $('.wy-nav-content').append(msFotter.pcFootHTML);
            }
        }
        if(width>768&&oldWidth<768){
          msFotter.documentEvaluationFn();
          msFotter.jumpForumStatistics();
        }
        oldWidth=width;
    });
  };

  // 文档显示格式化
  function docsformatter() {
     // 解决公式显示问题
     let emList = $('p>em');
     if(emList && emList.length > 0) {
         for (let i = 0; i < emList.length; i++) {
             if (emList[i].parentNode && emList[i].parentNode.innerHTML.indexOf('$') != -1) {
                 emList[i].parentNode.innerHTML = emList[i].parentNode.innerText;
             }
         }
     }

     // 说明样式调整
     if (pathname && pathname.indexOf('/design/introduction') != -1) {
         let blockquoteList = $('blockquote');
         blockquoteList.addClass('noteStyle');
     }

     //隐藏内容里面的toc
     // 当p标签里有文本时，不隐藏；仅存在a标签时才隐藏
     let accessable = true;
     $('.section>h1').siblings('ul.simple').first().find('li>p').each(function (idx, pElm) {
         let obj = $(pElm).clone();
         obj.find(':nth-child(n)').remove();
         if (obj.text().length) {
             accessable = false;
         }
         if (!accessable) {
             return;
         }
     });
     if (accessable) {
        $('.section>h1').siblings('ul.simple').first().css('display', 'none');
     }
     function resolveText(text) {
      return isEn ? 'Search in ' + text : '"' + text + '" 内搜索';
    }

    // 左侧文档菜单控制
    const wyMenu = $('.wy-grid-for-nav .wy-menu');
    if(!wyMenu.find('.notoctree-l2.current .current').next().length) {
      wyMenu.find('.notoctree-l2.current .current').append('<style>.wy-grid-for-nav .wy-menu .notoctree-l2.current .current::before{content:\'\'}</style>');
    }

    wyMenu.find('.caption').append('<img src=\'/pic/arrow-right.svg\' />').next().hide();

    if(wyMenu.find('.current').length) {
        $('.wy-grid-for-nav .wy-menu>.current').show().prev().addClass('down');
    } else {
      wyMenu.find('.caption').eq(0).addClass('down').next().show();
    }
    wyMenu.find('.caption').click(function () {
        $(this).toggleClass('down');
        $(this).next().toggle(200);
    });

    // 左侧菜单少于4个 展开显示
    if(wyMenu.find('.caption').length < 5) {
      wyMenu.find('.caption').each(function (index, item) {
          $(item).addClass('down').next().show();
      });
    }

     // 进入页面调整到锚点,解决中文锚点问题，中文锚点需要转码
    function gotoId() {
      let url = window.location.toString(); //进这个页面的url
      let id = window.decodeURIComponent(url.split('#')[1]); //中文id需要转码，英文id走catch error
      if (id) {
          if (document.querySelector(`#${id}`) !== null) {
              document.querySelector(`#${id}`).scrollIntoView(true);
          }
      }
    }

    setTimeout(function () {
        // 页面内搜索框提示具体搜索内容
        $('.wy-grid-for-nav #rtd-search-form input').eq(0).attr('placeholder', '"文档"内搜索');
        let welcomeText = isEn ? 'MindSpore Documentation': '欢迎查看MindSpore文档';
        if(pathname.startsWith('/tutorials/')){
          welcomeText = isEn ? 'MindSpore Tutorials': '欢迎查看MindSpore教程';
        }
        $('.wy-menu-vertical').before(
          `<div class="docsHome"><a  href="#" class="welcome">${welcomeText}</a></div>`
        );

        $('.welcome')[0].attributes[0].nodeValue = $('.icon-home')[0].attributes[0].nodeValue;

        var strTemp = $('header .selected').text();
        if(strTemp==''){
          strTemp = isEn ?'Docs':'文档';
        }
        $('#rtd-search-form input').attr('placeholder', resolveText(strTemp));

        gotoId();

    }, 100);

  }

  // 右侧锚点标识
  function sideRightAnchor() {
      let sectionList = $('.document>div:first-of-type>.section');
      let h1List = $('.section>h1');
      let h2List = $('.section>h2');
      let codeList = $('dl>dt>.descname:not(.method .descname)');
      if (sectionList[0] === undefined) {
          return;
      }
      let $ul =
          '<div class="navRight"><ul class="navList"><li class="navLi"><a href="#' +
          sectionList[0].id +
          '">' +
          h1List[0].innerText +
          '</a><ul class="navList2"></ul></li></ul></div>';
      let navLi3 = '',
          navLi2 = '',
          navLi4 = '';
      $('body').prepend($ul);
      if (h2List.length > 0) {
          for (let i = 0; i < h2List.length; i++) {
              // 正则去除括号、保留内容
              let id = h2List[i].parentNode.id
                  .replace(/\(([^).']*)\)/g, '$1')
                  .replace(/\“|\”|\'/g, '');
              let h3 = $('#' + id + ' ' + 'h3');
              if (h3.length > 0) {
                  navLi2 = '';
                  navLi3 = '';
                  for (let i = 0; i < h3.length; i++) {
                      if (
                          h3[i].parentNode.querySelectorAll('h4').length > 0
                      ) {
                          navLi4 = '';
                          let navLi4Array =
                              h3[i].parentNode.querySelectorAll('h4');
                          for (let k = 0; k < navLi4Array.length; k++) {
                              navLi4 +=
                                  '<li><a href=\'#' +
                                  navLi4Array[k].parentNode.id +
                                  '\'>' +
                                  navLi4Array[k].innerText +
                                  '</a></li>';
                          }
                          navLi3 +=
                              '<li><a href=\'#' +
                              h3[i].parentNode.id +
                              '\'>' +
                              h3[i].innerText +
                              '</a><ul class=\'navList4\'>' +
                              navLi4 +
                              '</ul></li>';
                      } else {
                          navLi3 +=
                              '<li><a href=\'#' +
                              h3[i].parentNode.id +
                              '\'>' +
                              h3[i].innerText +
                              '</a></li>';
                      }
                  }
                  navLi2 =
                      '<li><a href="#' +
                      h2List[i].parentNode.id +
                      '">' +
                      h2List[i].innerText +
                      '</a><ul class="navList3"></ul></li>';
              } else {
                  navLi3 = '';
                  navLi2 = '';
                  if (
                      h2List[i].parentNode.querySelectorAll(
                          '.class>dt .descname,.function>dt .descname'
                      ).length > 0
                  ) {
                      let navLi3Array = h2List[i].parentNode.querySelectorAll(
                          '.class>dt .descname,.function>dt .descname'
                      );

                      for (let j = 0; j < navLi3Array.length; j++) {
                          if (
                              navLi3Array[
                                  j
                              ].parentNode.parentNode.querySelectorAll(
                                  'dd .descname'
                              ).length > 0
                          ) {
                              navLi4 = '';
                              let navLi4Array =
                                  navLi3Array[
                                      j
                                  ].parentNode.parentNode.querySelectorAll(
                                      'dd .descname'
                                  );
                              for (let k = 0; k < navLi4Array.length; k++) {
                                  navLi4 +=
                                      '<li><a href=\'#' +
                                      navLi4Array[k].parentNode.id +
                                      '\'>' +
                                      navLi4Array[k].innerText +
                                      '</a></li>';
                              }
                              navLi3 +=
                                  '<li><a href=\'#' +
                                  navLi3Array[j].parentNode.id +
                                  '\'>' +
                                  navLi3Array[j].innerText +
                                  '</a><ul class="navList4">' +
                                  navLi4 +
                                  '</ul></li>';
                          } else {
                              navLi3 +=
                                  '<li><a href=\'#' +
                                  navLi3Array[j].parentNode.id +
                                  '\'>' +
                                  navLi3Array[j].innerText +
                                  '</a><ul class="navList4"></ul></li>';
                          }
                      }
                      navLi2 =
                          '<li><a href="#' +
                          h2List[i].parentNode.id +
                          '">' +
                          h2List[i].innerText +
                          '</a><ul class="navList3"></ul></li>';
                  } else {
                      navLi2 =
                          '<li><a href="#' +
                          h2List[i].parentNode.id +
                          '">' +
                          h2List[i].innerText +
                          '</a></li>';
                  }
              }

              $('.navList2').append(navLi2);
              $('.navList2>li:nth-of-type(' + (i + 1) + ') .navList3').append(
                  navLi3
              );
          }
      } else {
          navLi3 = '';
          for (let i = 0; i < codeList.length; i++) {
              var codeLi2 =
                  '<li><a href="#' +
                  codeList[i].parentNode.id +
                  '">' +
                  codeList[i].innerText +
                  '</a><ul class="navList3"></ul></li>';
              $('.navList2').append(codeLi2);
              navLi3 = '';
              if ($(codeList[i].parentNode).next().length) {
                  $(codeList[i].parentNode)
                      .next()
                      .find('.method .descname')
                      .each(function () {
                          navLi3 +=
                              '<li><a href="#' +
                              $(this)[0].parentNode.id +
                              '">' +
                              $(this).text() +
                              '</a></li>';
                      });
                  $('.navList2 .navList3').eq(i).append(navLi3);
              }
          }
      }

      // 点击右侧导航选中
      const navListLink = $('.navList a');
      navListLink.on('click', function () {
          navListLink.removeClass('selected');
          $(this).addClass('selected');
      });
      // 右侧导航可滚动蒙层
      function computeNavRightMask() {
          let scrollable =
              $('.navRight').height() < $('.navRight > .navList').height();
          let isScrolledAtBottom =
              $('.navRight > .navList').height() -
                  $('.navRight').scrollTop() <
              $('.navRight').height() + 10;
          if (scrollable && !isScrolledAtBottom) {
              $('.navRightMasker').css('display', 'block');
          } else {
              $('.navRightMasker').css('display', 'none');
          }
      }
      $('.navRight').wrap('<div class="navRightWraper"></div>');
      $('.navRightWraper').append('<div class="navRightMasker"></div>');
      computeNavRightMask();
      $('.navRight').scroll(computeNavRightMask);

      // 锚点跟随滚动定位\
      function navContentAnchor() {
          for (let i = 0; i < navListLink.length; i++) {
              let anchorId = navListLink.eq(i).attr('href').substring(1);
              let newAnchorId = escapeJquery(anchorId);
              if(!newAnchorId) return;
              if ($('#' + newAnchorId).offset().top - 60 < 116) {
                  navListLink.removeClass('selected');
                  navListLink.eq(i).addClass('selected');
              }
          }
          return false;
      }
      navContentAnchor();
      $('.wy-grid-for-nav').scroll(navContentAnchor);
  }

  const initPage = async function(){
    createScriptSensor();
    createScriptCommonJs();
    createScriptBaidu();

    watchWinResize();
    docsformatter();

    //获取导航菜单json
    headerMenuData = await getHeaderData(`/menu_${isEn?'en':'zh-CN'}.json`);
    msDocsVersion = await getHeaderData('/msVersion.json');

    body.prepend(msHeader.pcHeader);
    msHeader.headerMethods();

    $('.wy-nav-content').append(msFotter.pcFootHTML);
    msFotter.jumpForumStatistics();
    msFotter.documentEvaluationFn();

    // H5 显示
    isH5(() => {
      isH5Show();
    });

    $('.wy-breadcrumbs>li:first-of-type')[0].innerText = msHeader.pageTitle() + '(' + getCurrentVersion() + ')';

    sensorsMethods.getSearchKey();
    sideRightAnchor();
  };

  initPage();
});



















