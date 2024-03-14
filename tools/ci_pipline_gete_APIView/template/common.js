$(function () {
  (function () {
    $('title').text('MindSpore');
    const pathname = window.location.pathname;
    const isEn = pathname.includes('/en/');
    const lang = isEn ? '/en/' : '/zh-CN/';
    const enPath = isEn ? 'en' : '';
    const pathPrefix = pathname.split(lang);
    const currentVersion = pathPrefix[1].split('/')[0];
    const pagePath = pathPrefix[0] + lang + currentVersion;
    const body = $('body');
    let isDark = localStorage.getItem('ms-theme') === 'dark';
    // 隐私版本号
    const privacyVersion = '20240129';

    // 公共方法
    const utils = {
      newNavPath: isEn
        ? pathname.replace('/en/', '/zh-CN/')
        : pathname.replace('/zh-CN/', '/en/'),
      getHeaderMenu: [],
      getAllComponentVersion: [],
      getVersionData: [],
      getSubMenuData: [],
      getPageTitle: '',
      componentVersionTitle: '',
      configIP: {},
      searchUrl: isEn ? '/search/en' : '/search',
      // 获取版本 不带r
      getVersion: (version = '') => {
        return version.startsWith('r') ? version.slice(1) : version;
      },
      // xss
      filterXSS: (val) => {
        return $('<div>').text(val).html();
      },
      // 数据请求事件
      getRequest: (url) => {
        return new Promise((resolve, reject) => {
          $.ajax({
            type: 'get',
            url,
            dataType: 'json',
            success: (res) => {
              resolve(res);
            },
            error: (e) => {
              reject(e);
            },
          });
        }).catch(() => {});
      },
      // 判断是否是移动端
      isPad: () => {
        const screen =
          document.documentElement.clientWidth || document.body.clientWidth;
        if (screen < 1200) {
          return true;
        }
        return false;
      },
      // 弹窗
      dialogInit: (title, showClose = false) => {
        return `
          <div class="o-layer-dialog">
          <div class="o-layer-mask"></div>
          <div class="o-dlg-main">
            <div class="o-dlg-header">${title} ${
          showClose ? '<em class="o-dialog-closed"></em>' : ''
        }
            </div>
            <div class="o-scroller-container" id="dialog-content"></div>
          </div>
          </div>
        `;
      },
      destroyDialog: () => {
        $('.o-layer-dialog').remove();
      },
    };

    // header公共方法
    const appHeaderUtils = {
      // tag：['outlink','new','hot'....]
      getLinkTags: (tags) => {
        return `${
          tags && tags.length > 0
            ? tags
                .map((item) => {
                  if (item.toLocaleLowerCase() === 'outlink') {
                    return `<em class="outlink"></em>`;
                  } else {
                    return `<span class="ms-tag ${item.toLocaleLowerCase()}">
                            <span class="ms-tag-label">
                            ${utils.filterXSS(item)}
                            </span>
                        </span>`;
                  }
                })
                .join('')
            : ''
        }`;
      },
      // 代码仓
      getCodeContent: (className = '') => {
        const codeList = [
          {
            value: 'gitee',
            label: 'Gitee',
            href: `${utils.configIP.GITEE_URL}/mindspore/mindspore`,
          },
          {
            value: 'github',
            label: 'Github',
            href: `${utils.configIP.GITHUB_URL}/mindspore-ai/mindspore`,
          },
        ];
        return `
          ${codeList
            .map((item) => {
              return `<li class="${className}"><a href="${item.href}"  target="_blank" rel="noopener noreferrer">
                  ${item.label}
                  <em class="outlink"></em></a></li>`;
            })
            .join('')}
          `;
      },
      // 获取组件最新版本
      getLatestVersion: (path) => {
        let version = 'master';
        utils.getAllComponentVersion &&
          utils.getAllComponentVersion.forEach((item) => {
            if (path === item.name) {
              version =
                item.versions.length < 2
                  ? item.versions[0].version
                  : item.versions[1].version;
            } else if (path.startsWith(item.name) && item.eom !== 'true') {
              version =
                item.versions.length < 2
                  ? item.versions[0].version
                  : item.versions[1].version;
            }
          });
        return version;
      },
      // 获取组件版本链接
      getNavLinks: (path) => {
        let href = '';
        if (path === 'lite') {
          href = `/${path}${isEn ? '/en' : ''}`;
        } else if (path.startsWith('tutorials') || path === 'docs') {
          href = `/${path}${lang}${appHeaderUtils.getLatestVersion(
            path
          )}/index.html`;
        } else {
          href = `/${path}/docs${lang}${appHeaderUtils.getLatestVersion(
            path
          )}/index.html`;
        }
        return utils.filterXSS(href);
      },
      // 获取组件切换下拉列表
      getVersionsMenu: () => {
        let list = [];
        const matchingItem = utils.getAllComponentVersion.find(
          (item) => pathname.startsWith(`/${item.name}/`) && item.eom !== 'true'
        );
        if (matchingItem) {
          list = matchingItem.versions.map((sub) => ({
            version: utils.getVersion(sub.version),
            href:
              sub.href !== ''
                ? sub.href
                : `${pagePath.replace(currentVersion, sub.version)}/index.html`,
            versionAlias: utils.getVersion(sub.versionAlias),
          }));
        }
        return list.slice(0, 4);
      },
    };

    // header 移动端方法
    const appHeaderMb = {
      getMenuContentMb: () => {
        return `
        <div class="header-menu-layer">
        <div class="menu-mask"></div>
        <div class="menu-content">
        <div class="menu-left">
          <ul class="menu-list">
          <li class="menu-item"><a href="/" >${isEn ? 'Home' : '首页'}</a></li>
          ${
            utils.getHeaderMenu &&
            utils.getHeaderMenu
              .map((item, index) => {
                const href = item.href
                  ? isEn
                    ? item.href.en
                    : item.href.zh
                  : 'javascript:;';
                const name = isEn ? item.label.en : item.label.zh;
                return `
              <li class="menu-item">
                <a href="${utils.filterXSS(
                  href
                )}" data-val='${index}' rel="noopener noreferrer" target="${
                  item.jumOut ? '_blank' : '_self'
                }">
                      ${utils.filterXSS(name)}
                      </a>
              </li> `;
              })
              .join('')
          }
          </ul>
          <div class="menu-tool">
            <div class="menu-item"><a href="javascript:;" data-val='7'>${
              isEn ? 'Code' : '代码'
            }</a></div>
            <div class="tool-item">
            <a href="${utils.filterXSS(
              utils.newNavPath
            )}" class="dropdown-toggle" id="dropdownMenu1" data-toggle="dropdown" aria-haspopup="true" aria-expanded="true">
                    <span class="languageIpt">EN</span>
                </a>
                <div class="theme-change"><i class="icon-theme light"></i></div>
            </div>
          </div>
        </div>
        <div class="menu-right">
            ${
              utils.getHeaderMenu &&
              utils.getHeaderMenu
                .map((item) => {
                  return `<ul class="sub-menu">
                      ${appHeader.getHeaderNavLinks(
                        item.children,
                        'sub-menu-item'
                      )}
                </ul>`;
                })
                .join('')
            }
            <ul class="sub-menu">
              ${appHeaderUtils.getCodeContent('sub-menu-item')}
            </ul>
        </div>
        </div>
        </div>`;
      },
      // 移动端导航
      getHeaderContentMb: () => {
        return `<header id="header-h5" class="header-mobile">
        <div class="header-mobile-top">
        <div class="header-mobile-wrap">
        <div class="header-menu"> <em class="header-menu-icon"></em>
        ${appHeaderMb.getMenuContentMb()}
        </div>
        <span class="line"></span>
        <a href="/${enPath}"><img class="h5-logo" src="/pic/logo1.png" alt="logo" /></a>
        <p class="page-title">${utils.getPageTitle}</p></div>
        <em class="search-icon"></em>
        </div>
        </header>`;
      },
      // 移动端子菜单
      getNavContentMb: () => {
        return `<div id="nav-h5">
        <div class="header-mobile-menu ${
          utils.getSubMenuData.length > 0 ? 'current' : ''
        }">
          ${appHeader.getSubMenuContent()}
          </div>
          <div class="header-mobile-nav">
          <em class="page-menu"></em>
          </div>
          ${
            !pathname.startsWith('/tutorials')
              ? `<div class="docs-menu-icon h5"></div>`
              : ''
          }</div>
      `;
      },
      // 移动端交互
      utils: () => {
        $('.search-icon').on('click', function () {
          window.location.href = utils.searchUrl;
        });
        // 移动端菜单点击事件
        $('.header-menu-icon').on('click', function () {
          appHeader.toggleSideMenu(false);
          if ($(this).hasClass('on')) {
            $('.header-menu-layer').hide();
            $(this).removeClass('on');
          } else {
            $('.header-menu-layer').show();
            $(this).addClass('on');
          }
          $('.menu-left').find('.menu-item').removeClass('on');
        });
        $('.menu-mask').on('click', function () {
          $('.header-menu-layer').hide();
          $('.header-menu-icon').removeClass('on');
        });
        $('.menu-item a').on('click', function () {
          const menuIndex = $(this).data('val');
          $(this).closest('.menu-left').find('.menu-item').removeClass('on');
          $(this).parent().addClass('on');
          $('.menu-right .sub-menu').hide().eq(menuIndex).show();
        });

        // 国际化
        $('#dropdownMenu_h5').on('click', function () {
          $('.dropdown-menu').slideToggle();
        });

        // 侧栏导航显示
        $('.page-menu').on('click', function () {
          const isFlag = $(this).hasClass('on');
          appHeader.toggleSideMenu(!isFlag);
        });

        $('.h5.docs-menu-icon').on('click', function () {
          const title = isEn ? 'Document Directory' : '文档目录';
          body.prepend(utils.dialogInit(title));
          $('#mask').show();
          $('#dialog-content').append($('.header-nav-content').html());
        });

        $('.header-mobile-nav .versionText').on('click', function () {
          const title = isEn ? 'Select Version' : '选择版本';
          body.prepend(utils.dialogInit(title));
          $('#mask').show();
          $('#dialog-content').append($(this).siblings('.version-box').html());
        });

        // mask点击关闭侧栏
        $('#mask').on('click', function () {
          appHeader.toggleSideMenu(false);
          utils.destroyDialog();
          $(this).hide();
        });
      },
      init: () => {
        $('.wy-nav-content').append(
          '<div id="footer-mo"></div><div id="mask"></div>'
        );
        $('#footer').remove();
        body.prepend(appHeaderMb.getHeaderContentMb());
        $('.wy-nav-content-wrap').prepend(appHeaderMb.getNavContentMb());
        $('.header-mobile-nav').append(appHeader.getVersionList());
        $('#footer-mo').append(appFooter.getFooterContent());
        appHeaderMb.utils();
      },
    };

    // 初始化header
    const appHeader = {
      getSubMenuContent: () => {
        return `${
          utils.getSubMenuData &&
          utils.getSubMenuData
            .map((item) => {
              if (
                item.url.startsWith(pagePath) &&
                !item.url.includes('use/downloads')
              ) {
                item.active = 1;
              }
              return `<div class="header-nav-link">
              <a class="${
                item.active ? 'selected' : ''
              }" href="${utils.filterXSS(item.url)}">${utils.filterXSS(
                item.label
              )}</a>
            </div>
            `;
            })
            .join('')
        }`;
      },
      getHeaderContentInit: () => {
        return `<header class="header page-load">
        <nav class="header-wapper header-wapper-top">
        <div class="header-nav" style="display: flex;">
        <a href="/${enPath}" class="logo " >
        <img class="logo-img" src="/pic/logo_black.png" alt="logo" />
        </a>
        ${
          utils.getHeaderMenu &&
          utils.getHeaderMenu
            .map((item) => {
              const href = item.href
                ? isEn
                  ? item.href.en
                  : item.href.zh
                : '';
              const name = isEn ? item.label.en : item.label.zh;
              if (!item.children) {
                if (pathname.includes(item.id) && item.id === 'docs') {
                  item.active = 1;
                }
                return `
                  <div class="header-nav-link"><a class="header-nav-link-line ${
                    item.active ? 'selected' : ''
                  }" href="${utils.filterXSS(href)}">${utils.filterXSS(
                  name
                )}</a></div>
                  `;
              } else {
                if (
                  pathname.startsWith('/tutorials/') &&
                  item.id === 'learning'
                ) {
                  item.active = 1;
                }
                let navLabel = '';
                if (href) {
                  navLabel = `<a class="header-nav-link-line ${
                    item.active ? 'selected' : ''
                  }" href="${utils.filterXSS(
                    href
                  )}" rel="noopener noreferrer" target="${
                    item.jumOut ? '_blank' : '_self'
                  }">${utils.filterXSS(name)}</a>`;
                } else {
                  navLabel = `<span class="header-nav-link-line ${
                    item.active ? 'selected' : ''
                  }">${utils.filterXSS(name)}</span>`;
                }

                return `
                  <div class="header-nav-link">
                          <div class="nav-content">${navLabel}
                          ${
                            item.label && item.label[isEn ? 'en' : 'zh']
                              ? `<ul class="dropdown-menu-git ${
                                  item.id === 'docs' ? 'dropdown-menu-docs' : ''
                                }" >
                                ${appHeader.getHeaderNavLinks(
                                  item.children,
                                  ''
                                )}
                            </ul>`
                              : ''
                          }</div> ${appHeaderUtils.getLinkTags(item.tags)}
                  </div>`;
              }
            })
            .join('')
        }
        </div>
        <div class="header-nav navbar-tools" >
        <div class="header-search"><div class="search-input"><span class="search-icon"></span></span><input
          class="search-val" placeholder="${
            isEn ? 'Search...' : '全局搜索...'
          }"><span class="close-icon"></div></div>
        <div class="header-nav-link">
        <p class="code">${
          isEn ? 'Code' : '代码'
        } <i class="icon-chevron-down"></i></p>
        <ul class="dropdown-menu-git">
        ${appHeaderUtils.getCodeContent('')}
        </ul>
        </div>
        <div class="dropdown">
        <a href="${utils.filterXSS(
          utils.newNavPath
        )}" class="dropdown-toggle" id="dropdownMenu2" data-toggle="dropdown" aria-haspopup="true" aria-expanded="true">
          <span class="languageIpt">EN</span>
        </a>
        </div>
        <div class="theme-change"><i class="icon-theme light"></i></div>
        </div>
        </nav>
        <div class="header-menu">
        <nav class="header-wapper  header-wapper-docs" >
        <div class="header-nav">
        <div class="header-nav-info">
        ${
          !pathname.startsWith('/tutorials')
            ? `<em class="docs-menu-icon"></em>`
            : ''
        }
        <h3>${utils.getPageTitle}</h3>
        </div>
        <div class="bottom">
        ${appHeader.getSubMenuContent()}
        </div>
        </div>
        </div></nav></div><div class="header-nav-layer">
        <div class="header-nav-content">
        ${
          !pathname.startsWith('/tutorials')
            ? utils.getDocsMenu &&
              utils.getDocsMenu
                .map((item) => {
                  return `<div class="docsNew-column">
                      <div class="docsNew-column-title">${
                        utils.filterXSS(isEn ? item.title.en : item.title.zh) ||
                        ''
                      }</div>
                  <div class="bottom">
                      ${item.children
                        .map((subitem) => {
                          return `
                            <a class="docs-column-link" href="${
                              subitem.href
                                ? subitem.href
                                : appHeaderUtils.getNavLinks(subitem.id)
                            }" ${
                            subitem.link
                              ? 'target="_blank" rel="noopener noreferrer"'
                              : ''
                          }>
                            ${utils.filterXSS(subitem.name)}
                            ${appHeaderUtils.getLinkTags(subitem.tags)}
                          </a>`;
                        })
                        .join('')}
                  </div>
              </div>`;
                })
                .join('')
            : ''
        }
        </div></header>`;
      },
      // 导航子菜单
      getHeaderNavLinks: (data, className) => {
        return data
          ? data
              .map((sub) => {
                let list = '';
                if (sub.label && sub.label[isEn ? 'en' : 'zh']) {
                  let navHref = isEn ? sub.href.en : sub.href.zh;
                  if (sub.id === 'tutorials') {
                    navHref = navHref.replace(
                      'master',
                      appHeaderUtils.getLatestVersion(sub.id)
                    );
                  }
                  list = `<li class="${className}">
                      <a target="${
                        sub.jumOut ? '_blank' : '_self'
                      }" rel="noopener noreferrer" href="${utils.filterXSS(
                    navHref
                  )}">
                      ${utils.filterXSS(isEn ? sub.label.en : sub.label.zh)}
                      ${appHeaderUtils.getLinkTags(sub.tags)}
                      </a>
                    </li>`;
                }
                return list;
              })
              .join('')
          : '';
      },
      // 文档页面交互
      utils: () => {
        // 点击切换语言开始
        if (isEn) {
          $('.languageIpt').text('中');
          $('.logo').attr('href', '/en');
        } else {
          $('.languageIpt').text('EN');
        }
        // 点击关闭搜索框
        $('.close-icon').on('click', function () {
          $('.search-val').val('');
          $(this).hide();
        });
        // 获取搜索框的值并传递到搜索页面
        const searchInput = $('.search-input');
        $('.search-val')
          .focus(() => {
            searchInput.addClass('current');
          })
          .blur(() => {
            searchInput.removeClass('current');
          })
          .on('keydown', function (e) {
            const val = $(this).val();
            if (val !== '') {
              $('.close-icon').show();
            }
            if (e.keyCode === 13 && val !== '') {
              window.location.href =
                utils.searchUrl + '?inputValue=' + encodeURIComponent(val);
            }
          });

        // 文档菜单
        let timer = '';
        const headerLayer = $('.header-nav-layer');
        const docsMenuIcon = $('.docs-menu-icon');
        docsMenuIcon
          .mouseenter(() => {
            clearTimeout(timer);
            headerLayer.addClass('nav-show');
          })
          .mouseleave(() => {
            clearTimeout(timer);
            timer = setTimeout(() => {
              headerLayer.removeClass('nav-show');
            }, 300);
          });
        headerLayer
          .mouseenter(() => {
            clearTimeout(timer);
            headerLayer.addClass('nav-show');
          })
          .mouseleave(() => {
            headerLayer.removeClass('nav-show');
          });
        setTimeout(() => {
          $('.wy-nav-side').addClass('page-load');
        }, 50);
        setTimeout(() => {
          $('.header').addClass('page-load');
        }, 150);
      },
      // 组件切换列表
      getVersionList: () => {
        return (
          appHeaderUtils.getVersionsMenu().length > 0 &&
          `<div class='version-select-wrap'><div class="version-select-dom">
            <span class="versionText">
            ${utils.componentVersionTitle}
            <em class="icon-chevron-down"></em>
            </span>
              <div class="version-box"><ul class="version-list">
                    ${appHeaderUtils
                      .getVersionsMenu()
                      .map((item) => {
                        return `<li><a href="${utils.filterXSS(
                          item.href
                        )}" class='version-option'>${utils.filterXSS(
                          item.versionAlias === ''
                            ? item.version
                            : item.versionAlias
                        )}</a></li>`;
                      })
                      .join('')}
                    <li><a href="/versions${
                      isEn ? '/en' : ''
                    }" class='version-option'>${isEn ? 'More' : '更多'}</a></li>
                </ul></div>
            </div></div>`
        );
      },
      // 导航切换
      toggleSideMenu: (show) => {
        const wyNavSide = $('.wy-nav-side');
        const mask = $('#mask');
        const pageMenu = $('.page-menu');
        if (show) {
          wyNavSide.css({ left: '0' }).show();
          mask.css('zIndex', '999');
          pageMenu.addClass('on');
          body.addClass('overflow');
          mask.show();
        } else {
          wyNavSide.css({ left: '-90%' }).show();
          pageMenu.removeClass('on');
          body.removeClass('overflow');
          mask.hide();
        }
      },
      // 初始化菜单
      init: () => {
        body.prepend(appHeader.getHeaderContentInit());
        appHeader.utils();
        $('.header-nav-info').append(appHeader.getVersionList());
        docsUtils.init();
      },
    };

    // 初始化footer
    const appFooter = {
      beian: ['粤A2-20044005号', '粤公网安备44030702002890号'],
      aboutTitle: isEn ? 'Follow us' : '关注我们',
      copyRight: isEn ? 'Copyright©MindSpore 2024' : '版权所有©MindSpore 2024',
      footLinks: [
        {
          name: isEn ? 'Security' : '安全',
          path: `/security${isEn ? '/en' : ''}`,
        },
        {
          name: isEn ? 'Terms of Use' : '法律声明',
          path: `/legal${isEn ? '/en' : ''}`,
        },
        {
          name: isEn ? 'Privacy Policy' : '隐私政策',
          path: `/privacy${isEn ? '/en' : ''}`,
        },
      ],
      // PC footer
      getFooterContent: () => {
        return `
          <div id="footer">
          <div class="footer-bottom">
          <div class="copyright">
          <span class="copyRight">${appFooter.copyRight}</span>
          <a class="copynum" rel="noopener noreferrer" target="_blank" href="${
            utils.configIP.BEIAN_URL
          }">${appFooter.beian[0]}</a>
          <div>
          <a class="footer-record" rel="noopener noreferrer" target="_blank" href="${
            utils.configIP.BEIAN_MPS_URL
          }">
          <img class="copyImg2" src="/pic/copyright3.png" alt="img" />
          <span class="keepRecord">${appFooter.beian[1]} </span>
          </a>
          </div>
          </div>
          <div class="footer-menu">
          ${appFooter.footLinks
            .map((item) => {
              return `<a href="${item.path}">${item.name}</a>`;
            })
            .join('')}
          </div>
          </div>
        `;
      },
      // 初始化
      init: () => {
        $('.wy-nav-content').append(appFooter.getFooterContent());
      },
    };

    // 文档方法
    const docsUtils = {
      // 用一个宽度来记录之前的宽度，当之前的宽度和现在的宽度，从手机切换到电脑或者从电脑切换到手机，才执行下面部分函数
      watchWinResize: () => {
        window.addEventListener('resize', () => {
          const width = window.innerWidth;
          const h5Head = document.getElementById('header-h5');
          const h5footer = document.getElementById('footer-mo');
          const pcfooter = document.getElementById('footer');
          const navh5 = document.getElementById('nav-h5');
          if (width < 1200) {
            $('#mask').css('display', 'none');
            $('.wy-nav-side').css({ left: '-90%', transition: '0s' });
            if (h5Head === null) {
              appHeaderMb.init();
            }
            utils.destroyDialog();
          } else {
            $('.header').css('display', 'block');
            $('#mask').remove();
            $('.wy-nav-side').css('left', '0');

            if (h5Head || h5footer || navh5) {
              h5Head.remove();
              h5footer.remove();
              navh5.remove();
            }

            if (pcfooter === null) {
              $('.wy-nav-content').append(appFooter.getFooterContent());
            }
          }
        });
      },
      // 复制粘贴功能
      onClipboard: () => {
        $('pre>span:first-of-type').append(
          '<em class="copy-btn" data-tooltip="copy"><i class="icon-copy"></i></em>'
        );

        $('.copy-btn').click(function () {
          const that = $(this);
          that.attr('data-tooltip', isEn ? 'Copy Success!' : '复制成功！');
          const domTemp = that.parent().parent()[0].cloneNode(true);
          domTemp.querySelectorAll('.go').forEach((item) => item.remove());
          let Url2 = domTemp.innerText.replace('Copy', '');
          Url2 = Url2.split(/[\n]/);
          // 只有python才需要去掉>>>和...
          const parentElement = that
            .closest('.highlight-cpp, .highlight-c++, .highlight-java')
            .get(0);

          if (parentElement) {
            Url2 = Url2.join('\n');
          } else {
            let flag = false;
            for (let i = 0; i < Url2.length; i++) {
              if (Url2[i].includes('>>>') || Url2[i].indexOf('...') === 0) {
                flag = true;
              } else {
                flag = false;
              }
              if (flag) {
                if (Url2[i].includes('>>>') || Url2[i].indexOf('...') === 0) {
                  Url2[i] = Url2[i].slice(4);
                } else {
                  Url2[i] = '';
                }
              }
            }
            Url2 = Url2.join('\n').replace(/>>> /g, '').replace(/>>>/g, '');
          }
          const oInput = document.createElement('textarea');
          oInput.value = Url2;
          document.body.appendChild(oInput);
          oInput.select(); // 选择对象
          document.execCommand('Copy'); // 执行浏览器复制命令
          oInput.class = 'oInput';
          oInput.style.display = 'none';
          setTimeout(() => {
            that.attr('data-tooltip', 'copy');
          }, 1000);
        });
      },
      // 文档反馈
      getFeedback: () => {
        let s = document.getElementsByTagName('HEAD')[0];
        let hm = document.createElement('script');
        hm.src = '/feedback.js';
        s.appendChild(hm, s);
      },
      // 文档反馈
      getFeedbackContent: () => {
        const askQuestion = isEn ? 'Document Feedback' : '文档反馈';
        const askQuestion1 = isEn ? 'Quick Feedback' : '快速反馈问题';
        const askQuestionInfo = isEn
          ? 'Click the blue button to fill in document feedback information and go to the code repository to submit an issue or PR. We will follow up on it.'
          : '点击蓝色按钮，可填写文档反馈相关信息，并跳转代码仓提issue或PR，我们将会跟进处理';
        const askQuestionInfo1 = isEn
          ? 'Remember to add the tag below:'
          : '记得添加mindspore-assistant标签哦！';

        const feedbackDom = `<div class="docs-feedback">
          <div class="feedback-box ${isEn ? 'en' : ''}">
          <span class="text askQuestion">${askQuestion}</span>
          <div class="feedback-layer">
            <p class="title"><i class="feedback-icon"></i>${askQuestion1}</p>
            <p class="desc">${askQuestionInfo}</p>
            <p class="desc m12">${askQuestionInfo1}</p>
            <p class="desc assistant">mindspore- assistant</p>
          </div>
          </div>
          <div class="go-top"><i class="icon-gotop"></i></div>
          </div>`;

        body.prepend(feedbackDom);
        $('.docs-feedback .go-top').on('click', () => {
          $('.wy-nav-content-wrap').animate({ scrollTop: 0 }, 300);
        });
      },
      init: () => {
        docsUtils.onClipboard();
        docsUtils.watchWinResize();
        docsUtils.getFeedbackContent();
        docsUtils.getFeedback();
        docsAnchor.init();

        // 解决公式显示问题
        const emList = $('p>em');
        if (emList && emList.length > 0) {
          for (let i = 0; i < emList.length; i++) {
            if (
              emList[i].parentNode &&
              emList[i].parentNode.innerHTML.includes('$')
            ) {
              emList[i].parentNode.innerHTML = emList[i].parentNode.innerText;
            }
          }
        }

        // 说明样式调整
        if (pathname && pathname.includes('/design/introduction')) {
          const blockquoteList = $('blockquote');
          blockquoteList.addClass('noteStyle');
        }

        // 左侧菜单控制
        const wyMenu = $('.wy-grid-for-nav .wy-menu');
        wyMenu
          .find('.caption')
          .append("<i class='icon-chevron-down'></i>")
          .next()
          .hide();

        if (wyMenu.find('.current').length) {
          $('.wy-grid-for-nav .wy-menu>.current')
            .show()
            .prev()
            .addClass('down');
        } else {
          wyMenu.find('.caption').eq(0).addClass('down').next().show();
        }
        wyMenu.find('.caption').click(function () {
          $(this).toggleClass('down').next().toggle(200);
        });

        // 左侧菜单少于4个 展开显示
        if (wyMenu.find('.caption').length < 5) {
          wyMenu.find('.caption').each(function (item) {
            $(item).addClass('down').next().show();
          });
        }
        // 默认展开文档内容
        if (
          pathname.startsWith('/docs/zh-CN/') ||
          pathname.startsWith('/docs/en/')
        ) {
          if (
            pathname.includes('/index.html') ||
            pathname.includes('/search.html') ||
            pathname.includes('/_modules/')
          ) {
            wyMenu.find('.caption').removeClass('down').next().hide();
            wyMenu.find('.caption').eq(2).addClass('down').next().show();
          }
        }

        // 进入页面调整到锚点,解决中文锚点问题，中文锚点需要转码
        function gotoId() {
          const url = utils.filterXSS(window.location.toString()); // 进这个页面的url
          const id = window.decodeURIComponent(url.split('#')[1]); // 中文id需要转码，英文id走catch error
          if (id && document.getElementById(id) !== null) {
            document.getElementById(id).scrollIntoView(true);
          }
        }
        gotoId();

        const resolveText = (text) => {
          return isEn ? `Search in ${text} ` : `"${text}" 内搜索`;
        };

        let strTemp = isEn ? 'Docs' : '文档';
        let docsHome = isEn
          ? `${utils.getPageTitle} Documentation`
          : `欢迎查看${utils.getPageTitle}文档`;

        if (pathname.startsWith('/tutorials')) {
          strTemp = isEn ? 'Tutorials' : '教程';
          docsHome = isEn
            ? `MindSpore ${utils.getPageTitle}`
            : `欢迎查看MindSpore${utils.getPageTitle}`;
        }
        $('#rtd-search-form input').attr(
          'placeholder',
          utils.filterXSS(resolveText(strTemp))
        );

        // 返回组件index
        $('.wy-side-nav-search').after(
          `<div class="docs-home"><a href="${pagePath}/index.html" class="welcome">${docsHome}</a></div>`
        );

        // lite 配置
        if (
          pathname.startsWith('/lite/docs') &&
          pathname.includes('/use/downloads')
        ) {
          $('.header-wapper-docs .bottom .header-nav-link a')
            .removeClass('selected')
            .eq(1)
            .addClass('selected');
          $('.header-nav-info .version-select-wrap').hide();
        }

        $('.wy-breadcrumbs').parent('div').addClass('rst-content-top');
        $('.wy-breadcrumbs>li:first-of-type')[0].innerText =
          utils.getPageTitle + ' (' + utils.componentVersionTitle + ')';
      },
    };

    // 文档右侧锚点
    const docsAnchor = {
      getNodeList: (
        node,
        isShow = false,
        isId = false,
        child = '',
        className = ''
      ) => {
        const name = utils.filterXSS(node.innerText);
        try {
          return `<li>
          <span class="line"></span>
          <a title="${name}" href="#${utils.filterXSS(
            isId
              ? docsAnchor.getReplaceStr(node.closest('dt').id)
              : docsAnchor.getReplaceStr(node.parentNode.id)
          )}">${name}</a>
          ${
            isShow
              ? `<ul class='${utils.filterXSS(className)}'>${child || ''}</ul>`
              : ''
          }
        </li> `;
        } catch (error) {}
      },
      getReplaceStr: (str) => {
        return str
          .replace(/\(([^).']*)\)/g, '$1')
          .replace(/\“|\”|\'/g, '')
          .replace(/\!|\=|\:/g, '');
      },
      getAnchorList: () => {
        const sectionList = $('.document>div:first-of-type>section');
        const h1List = $('.document section>h1');
        const h2List = $('.document section>h2');
        const codeList = $(
          'dl>dt>.descname>.pre:not(.method .descname>.pre):not(.property .descname>.pre)'
        );
        if (sectionList[0] === undefined) {
          return;
        }
        const $ul = `<div class="navRightWraper"><div class="navRight">
            <ul class="navList">
              <li class="navLi"><a href="#${utils.filterXSS(
                sectionList[0].id
              )}" class="navLiTitle">${utils.filterXSS(h1List[0].innerText)}</a>
                <ul class="navList2"></ul>
              </li></ul>
          </div></div>`;
        let navLi3 = '';
        let navLi2 = '';
        let navLi4 = '';
        $('.wy-nav-content-wrap').append($ul);
        if (h2List.length > 0) {
          for (let i = 0; i < h2List.length; i++) {
            // 正则去除括号、保留内容
            const id = docsAnchor.getReplaceStr(h2List[i].parentNode.id);
            const h3 = document.getElementById(id).querySelectorAll('h3');
            if (h3.length > 0) {
              navLi2 = '';
              navLi3 = '';
              for (let i = 0; i < h3.length; i++) {
                if (h3[i].parentNode.querySelectorAll('h4').length > 0) {
                  navLi4 = '';
                  const navLi4Array = h3[i].parentNode.querySelectorAll('h4');
                  for (let k = 0; k < navLi4Array.length; k++) {
                    navLi4 += docsAnchor.getNodeList(navLi4Array[k]);
                  }
                  navLi3 += docsAnchor.getNodeList(
                    h3[i],
                    true,
                    false,
                    navLi4,
                    'navList4'
                  );
                } else {
                  navLi3 += docsAnchor.getNodeList(h3[i]);
                }
              }
              navLi2 = docsAnchor.getNodeList(
                h2List[i],
                true,
                false,
                '',
                'navList3'
              );
            } else {
              navLi3 = '';
              navLi2 = '';
              if (
                h2List[i].parentNode.querySelectorAll(
                  '.class>dt .descname>.pre,.function>dt .descname>.pre'
                ).length > 0
              ) {
                const navLi3Array = h2List[i].parentNode.querySelectorAll(
                  '.class>dt .descname>.pre,.function>dt .descname>.pre'
                );

                for (let j = 0; j < navLi3Array.length; j++) {
                  if (
                    navLi3Array[j].parentNode.parentNode.querySelectorAll(
                      'dd .pre'
                    ).length > 0
                  ) {
                    navLi4 = '';
                    const navLi4Array =
                      navLi3Array[j].parentNode.parentNode.querySelectorAll(
                        'dd .descname>.pre'
                      );
                    for (let k = 0; k < navLi4Array.length; k++) {
                      navLi4 += docsAnchor.getNodeList(
                        navLi4Array[k],
                        false,
                        true
                      );
                    }
                    navLi3 += docsAnchor.getNodeList(
                      navLi3Array[j],
                      true,
                      true,
                      navLi4,
                      'navList4'
                    );
                  } else {
                    navLi3 += docsAnchor.getNodeList(
                      navLi3Array[j],
                      true,
                      true,
                      '',
                      'navList4'
                    );
                  }
                }
                navLi2 = docsAnchor.getNodeList(
                  h2List[i],
                  true,
                  false,
                  '',
                  'navList3'
                );
              } else {
                navLi2 = docsAnchor.getNodeList(h2List[i]);
              }
            }

            $('.navList2').append(navLi2);
            $('.navList2>li:nth-of-type(' + (i + 1) + ') .navList3').append(
              navLi3
            );
          }
        } else {
          codeList.each(function () {
            let codeText = $(this).text();
            let parentID = $(this).closest('dt').attr('id');
            codeText = utils.filterXSS(codeText);
            parentID = utils.filterXSS(parentID);
            const codeLi2 = `
              <li>
              <a title="${codeText}" href="#${parentID}">${codeText}</a>
              <ul class="navList3"></ul>
              </li>`;
            $('.navList2').append(codeLi2);
            const nextDescnames = $(this)
              .closest('dt')
              .next()
              .find('.descname>.pre');
            if (nextDescnames.length) {
              const navLi3 = nextDescnames
                .map(function () {
                  const descnameText = $(this).text();
                  const descnameParentID = $(this).closest('dt').attr('id');
                  return `<li><a href="#${descnameParentID}" title="${descnameText}">${descnameText}</a></li>`;
                })
                .get()
                .join('');
              $('.navList2 .navList3').last().append(navLi3);
            }
          });
        }
      },
      utils: () => {
        const navListLink = $('.navList a');
        navListLink.on('click', function () {
          navListLink.closest('li').removeClass('selected');
          $(this).closest('li').addClass('selected');
        });
        const getCurrentSelected = () => {
          for (let i = 0; i < navListLink.length; i++) {
            const id = navListLink.eq(i).attr('href').substring(1);
            const newId = id.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
            try {
              if (!newId) return;
              if (
                $('#' + newId).length > 0 &&
                $('#' + newId).offset().top - 140 < 116
              ) {
                navListLink.closest('li').removeClass('selected');
                navListLink.eq(i).closest('li').addClass('selected');
              }
            } catch (error) {}
          }
          return false;
        };
        getCurrentSelected();
      },
      affix: () => {
        const scrollable = $('.wy-nav-content-wrap').scrollTop();
        if (scrollable > 90) {
          $('.navRightWraper').addClass('fixed');
        } else {
          $('.navRightWraper').removeClass('fixed');
        }
        docsAnchor.utils();
      },
      init: () => {
        docsAnchor.getAnchorList();
        $('.wy-nav-content-wrap').scroll(docsAnchor.affix);
      },
    };

    // 百度统计
    const getBaiduSensor = () => {
      const hm = document.createElement('script');
      hm.src = utils.configIP.BAIDU_HM;
      const s = document.getElementsByTagName('script')[0];
      s.parentNode.insertBefore(hm, s);
    };

    // 换肤
    const themeInit = () => {
      const themeStyle = localStorage.getItem('ms-theme');
      const themeIcon = $('.theme-change i');
      const documentElement = document.documentElement;
      themeIcon.removeClass('dark light');
      let logoImg = 'logo-zh-light.svg';

      if (!themeStyle) {
        localStorage.getItem('ms-theme', 'light');
        documentElement.removeAttribute('data-o-theme');
        themeIcon.addClass('light');
      } else {
        documentElement.setAttribute('data-o-theme', themeStyle);
        themeIcon.addClass(themeStyle);
      }
      isDark = themeStyle === 'dark';
      logoImg = isEn
        ? isDark
          ? 'logo-en-dark.svg'
          : 'logo-en-light.svg'
        : isDark
        ? 'logo-zh-dark.svg'
        : 'logo-zh-light.svg';

      $('.logo-img').attr('src', '/pic/' + logoImg);

      themeIcon.click(function () {
        let theme = 'light';
        if ($(this).hasClass('light')) {
          theme = 'dark';
          $(this).addClass('dark').removeClass('light');
          documentElement.setAttribute('data-o-theme', theme);
        } else {
          theme = 'light';
          $(this).addClass('light').removeClass('dark');
          documentElement.removeAttribute('data-o-theme');
        }
        localStorage.setItem('ms-theme', theme);
        isDark = theme === 'dark';
        logoImg = isEn
          ? isDark
            ? 'logo-en-dark.svg'
            : 'logo-en-light.svg'
          : isDark
          ? 'logo-zh-dark.svg'
          : 'logo-zh-light.svg';
        $('.logo-img').attr('src', '/pic/' + logoImg);
      });
    };

    // cookie管理
    const cookieNotice = {
      COOKEY_KEY: 'agreed-cookiepolicy',
      COOKIE_AGREED_STATUS: {
        NOT_SIGNED: '0', // 未签署
        ALL_AGREED: '1', // 同意所有cookie
        NECCESSARY_AGREED: '2', // 仅同意必要cookie
      },
      locale: {
        title: !isEn
          ? 'MindSpore社区重视您的隐私'
          : 'MindSpore Community Respects Your Privacy',
        desc: !isEn
          ? '我们在本网站上使用Cookie，包括第三方Cookie，以便网站正常运行和提升浏览体验。单击“全部接受”即表示您同意这些目的；单击“全部拒绝”即表示您拒绝非必要的Cookie；单击“管理Cookie”以选择接受或拒绝某些Cookie。需要了解更多信息或随时更改您的 Cookie 首选项，请参阅我们的 '
          : 'This site uses cookies from us and our partners to improve your browsing experience and make the site work properly. By clicking "Accept All", you consent to the use of cookies. By clicking "Reject All", you disable the use of unnecessary cookies. You can manage your cookie settings by clicking "Manage Cookies". For more information or to change your cookie settings, please refer to our',
        privacy: !isEn ? '《隐私政策》。' : 'Privacy Policy.',
        privacyHref: !isEn ? '/privacy' : '/privacy/en',
        action: [
          {
            btn: '全部接受',
            btnEn: 'Accept All',
            type: 'all',
          },
          {
            btn: '全部拒绝',
            btnEn: 'Reject All',
            type: 'refuse',
          },
          {
            btn: ' 管理Cookie ',
            btnEn: ' Manage Cookies ',
            type: 'manage',
          },
        ],
        manageTitle: !isEn ? '管理Cookie' : 'Manage Cookies',
        necessaryTitle: !isEn ? '必要Cookie' : 'Strictly Necessary Cookies',
        necessaryDesc: !isEn
          ? '这些Cookie是网站正常工作所必需的，不能在我们的系统中关闭。它们通常仅是为了响应您的服务请求而设置的，例如登录或填写表单。您可以将浏览器设置为阻止Cookie来拒绝这些Cookie，但网站的某些部分将无法正常工作。这些Cookie不存储任何个人身份信息。'
          : 'These cookies are necessary for the site to work properly and cannot be switched off. They are usually only set in response to actions made by you which amount to a request for services, such as logging in or filling in forms. You can set the browser to block these cookies, but that can make parts of the site not work. These cookies do not store any personally identifiable information.',
        statisticsTitle: !isEn ? '统计分析Cookie' : 'Analytics Cookies',
        statisticsDesc: !isEn
          ? '我们将根据您的同意使用和处理这些非必要Cookie。这些Cookie允许我们获得摘要统计数据，例如，统计访问量和访问者来源，便于我们改进我们的网站。'
          : 'We will use these cookies only with your consent. These cookies help us make improvements by collecting statistics such as the number of visits and traffic sources.',
        enabled: !isEn ? '始终启用' : 'Always active',
        manageAction: [
          {
            btn: '保存并接受',
            btnEn: 'Save and Accept',
            type: 'save',
          },
          {
            btn: '全部接受',
            btnEn: 'Accept All',
            type: 'allow-all',
          },
        ],
      },
      isNoticeVisible: true,
      // setCookie 设置cookie
      setCustomCookie: (cname, cvalue, day = 1) => {
        let expires = new Date(Date.now() + day * 864e5);
        if (expires) {
          expires = expires.toUTCString();
        }
        document.cookie = `${cname}=${cvalue};expires=${expires};path=/`;
      },
      // 获取cookie值
      getCookieByKey: (key) => {
        const cookieArr = document.cookie.split('; ');
        for (let i = 0, len = cookieArr.length; i < len; i++) {
          const item = cookieArr[i];
          const rlt = item.split('=');
          if (rlt[0] === key) {
            return rlt[1];
          }
        }
      },
      // 是否未签署
      isNotSigned: () => {
        return (
          cookieNotice.getUserCookieStatus() ===
          cookieNotice.COOKIE_AGREED_STATUS.NOT_SIGNED
        );
      },
      // 是否同意所有
      isAllAgreed: () => {
        return (
          cookieNotice.getUserCookieStatus() ===
          cookieNotice.COOKIE_AGREED_STATUS.ALL_AGREED
        );
      },
      // 弹框 是否选中统计分析Cookie
      isManageAgreed: () => {
        return $('.statistics-switch').prop('checked');
      },
      // 显示/隐藏cookie提示
      toggleNoticeVisible: (val) => {
        const cookieMain = $('.cookie-notice');
        val ? cookieMain.show() : cookieMain.hide();
      },
      // 获取cookie状态
      getUserCookieStatus: () => {
        const { COOKIE_AGREED_STATUS, COOKEY_KEY } = cookieNotice;
        const cookieVal = cookieNotice.getCookieByKey(COOKEY_KEY) ?? '0';

        const cookieStatusVal = cookieVal[0];
        const privacyVersionVal = cookieVal.slice(1);

        if (privacyVersionVal !== privacyVersion) {
          return COOKIE_AGREED_STATUS.NOT_SIGNED;
        }

        if (cookieStatusVal === COOKIE_AGREED_STATUS.ALL_AGREED) {
          return COOKIE_AGREED_STATUS.ALL_AGREED;
        } else if (cookieStatusVal === COOKIE_AGREED_STATUS.NECCESSARY_AGREED) {
          return COOKIE_AGREED_STATUS.NECCESSARY_AGREED;
        } else {
          return COOKIE_AGREED_STATUS.NOT_SIGNED;
        }
      },
      // cookie提示内容
      getCookieContent: () => {
        return `<div class="cookie-notice"><div class="cookie-notice-content">
              <div class="content-wrapper cookie-notice-wrap">
                <div class="cookie-notice-left">
                  <p class="cookie-title">${cookieNotice.locale.title}</p>
                  <p class="cookie-desc">
                    ${cookieNotice.locale.desc}
                    <a href="${
                      cookieNotice.locale.privacyHref
                    }" rel="noopener noreferrer" target="_blank">${
          cookieNotice.locale.privacy
        }</a>
                  </p>
                </div>
                <div class="cookie-notice-right">
                  ${cookieNotice.locale.action
                    .map((item) => {
                      return `<button class="o-button ${item.type}">${
                        isEn ? item.btnEn : item.btn
                      }</button>`;
                    })
                    .join('')}
                </div>
                <em class="cookie-close"></em>
              </div>
            </div></div>`;
      },
      // 弹框内容
      getManageContent: () => {
        return `
              <div class="manage-content">
                <div class="manage-item">
                  <div class="item-header">
                    <span class="item-title">${
                      cookieNotice.locale.necessaryTitle
                    }</span>
                    <span class="item-extra">${
                      cookieNotice.locale.enabled
                    }</span>
                  </div>
                  <p class="item-detail">
                    ${cookieNotice.locale.necessaryDesc}
                  </p>
                </div>
                <div class="manage-item">
                  <div class="item-header">
                    <span class="item-title">${
                      cookieNotice.locale.statisticsTitle
                    }</span>
                    <span class="item-extra">
                      <input type="checkbox" class="statistics-switch" is="o-switch">
                    </span>
                  </div>
                  <p class="item-detail">
                    ${cookieNotice.locale.statisticsDesc}
                  </p>
                </div>
                <div class="manage-action">
                ${cookieNotice.locale.manageAction
                  .map((item) => {
                    return `<button class="o-button ${item.type}">${
                      isEn ? item.btnEn : item.btn
                    }</button>`;
                  })
                  .join('')}
                </div>
              </div>`;
      },
      // 用户同意所有cookie
      acceptAll: () => {
        const {
          setCustomCookie,
          COOKEY_KEY,
          COOKIE_AGREED_STATUS,
          toggleNoticeVisible,
        } = cookieNotice;

        getBaiduSensor();
        setCustomCookie(
          COOKEY_KEY,
          `${COOKIE_AGREED_STATUS.ALL_AGREED}${privacyVersion}`,
          180
        );
        toggleNoticeVisible(false);
      },
      // 用户拒绝所有cookie，即仅同意必要cookie
      rejectAll: () => {
        const {
          setCustomCookie,
          COOKEY_KEY,
          COOKIE_AGREED_STATUS,
          toggleNoticeVisible,
        } = cookieNotice;

        setCustomCookie(
          COOKEY_KEY,
          `${COOKIE_AGREED_STATUS.NECCESSARY_AGREED}${privacyVersion}`,
          180
        );
        toggleNoticeVisible(false);
        utils.destroyDialog();
      },
      removeNotice: () => {
        $('.cookie-notice').remove();
      },
      utils: () => {
        $('.cookie-notice-right button').on('click', function () {
          const {
            locale,
            getManageContent,
            acceptAll,
            rejectAll,
            removeNotice,
          } = cookieNotice;
          // 同意
          if ($(this).hasClass('all')) {
            acceptAll();
            removeNotice();
          }
          // 拒绝
          if ($(this).hasClass('refuse')) {
            rejectAll();
            removeNotice();
          }
          // 管理cookie
          if ($(this).hasClass('manage')) {
            if (utils.isPad()) {
              $('#mask').show();
            }
            utils.destroyDialog();
            body.prepend(utils.dialogInit(locale.manageTitle, true));
            $('#dialog-content').append(getManageContent);
            cookieNotice.utils();
          }
        });
        // 弹框按钮事件
        $('.manage-action button').on('click', function () {
          const { acceptAll, rejectAll, isManageAgreed, removeNotice } =
            cookieNotice;
          // 保存设置
          if ($(this).hasClass('save')) {
            isManageAgreed() ? acceptAll() : rejectAll();
          }
          // 同意所有
          if ($(this).hasClass('allow-all')) {
            acceptAll();
            $('.statistics-switch').prop('checked', true);
          }
          removeNotice();
          utils.destroyDialog();
          if (utils.isPad()) {
            $('#mask').hide();
          }
        });
        // 关闭弹窗
        $('.o-layer-mask,.o-dialog-closed').on('click', function () {
          utils.destroyDialog();
          if (!cookieNotice.isAllAgreed()) {
            $('.statistics-switch').prop('checked', false);
          }
        });
        // 隐藏cookie
        $('.cookie-close').on('click', function () {
          cookieNotice.toggleNoticeVisible(false);
        });
      },
      init: () => {
        if (cookieNotice.isNotSigned()) {
          body.append(cookieNotice.getCookieContent());
          cookieNotice.utils();
          cookieNotice.toggleNoticeVisible(true);
        }

        if (cookieNotice.isAllAgreed()) {
          cookieNotice.acceptAll();
        }
      },
    };

    const initPage = async () => {
      // 获取导航菜单json
      utils.getHeaderMenu = await utils.getRequest(`/header.json`);
      // 获取文档导航菜单json
      utils.getDocsMenu = await utils.getRequest(`/docs-menu.json`);
      utils.getAllComponentVersion = await utils.getRequest('/ms-version.json');
      // 公网ip配置
      utils.configIP = await utils.getRequest('/config.json');

      // 页面信息
      utils.getVersionData = await utils.getRequest(
        `${pagePath}/_static/js/version.json`
      );

      const pageTitle = isEn
        ? utils.getVersionData.label.en
        : utils.getVersionData.label.zh;
      utils.getPageTitle = utils.filterXSS(pageTitle);

      // 子菜单
      if (utils.getVersionData.submenu) {
        utils.getSubMenuData = isEn
          ? utils.getVersionData.submenu.en
          : utils.getVersionData.submenu.zh;
      }

      // 页面组件版本
      const componentVersionTitle = utils.getVersionData.versionAlias
        ? utils.getVersionData.versionAlias
        : utils.getVersionData.version;
      utils.componentVersionTitle = utils.filterXSS(
        utils.getVersion(componentVersionTitle)
      );

      appHeader.init();
      appFooter.init();
      // cookie提示
      cookieNotice.init();

      if (utils.isPad()) {
        appHeaderMb.init();
      }

      themeInit();
    };
    initPage();
  })();
});
