define("discourse/plugins/docker_manager/discourse/components/docker-manager/console",["exports","@glimmer/component","@ember/render-modifiers/modifiers/did-insert","@ember/render-modifiers/modifiers/did-update","discourse/lib/decorators","@ember/component","@ember/template-factory"],(function(e,t,r,s,n,a,i){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class o extends t.default{scrollToBottom(e){this.args.followOutput&&(e.scrollTop=e.scrollHeight)}static#e=(()=>dt7948.n(this.prototype,"scrollToBottom",[n.bind]))()
static#t=(()=>(0,a.setComponentTemplate)((0,i.createTemplateFactory)({id:"rUSYTm9D",block:'[[[1,"\\n    "],[11,0],[24,0,"console-logs"],[4,[32,0],[[30,0,["scrollToBottom"]]],null],[4,[32,1],[[30,0,["scrollToBottom"]],[30,1]],null],[12],[1,[30,1]],[13],[1,"\\n  "]],["@output"],false,[]]',moduleName:"/var/www/discourse/app/assets/javascripts/discourse/discourse/plugins/docker_manager/discourse/components/docker-manager/console.js",scope:()=>[r.default,s.default],isStrictMode:!0}),this))()}e.default=o})),define("discourse/plugins/docker_manager/discourse/components/docker-manager/progress-bar",["exports","@glimmer/component","@ember/template","discourse/helpers/concat-class","@ember/component","@ember/template-factory"],(function(e,t,r,s,n,a){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class i extends t.default{get active(){return 100!==parseInt(this.args.percent,10)}get barStyle(){let e=parseInt(this.args.percent,10)
return e>100&&(e=100),(0,r.htmlSafe)(`width: ${e}%`)}static#e=(()=>(0,n.setComponentTemplate)((0,a.createTemplateFactory)({id:"jR8g5Q7t",block:'[[[1,"\\n    "],[10,0],[15,0,[28,[32,0],["progress-bar",[52,[30,0,["active"]],"active"]],null]],[12],[1,"\\n      "],[10,0],[14,0,"progress-bar-inner"],[15,5,[30,0,["barStyle"]]],[12],[13],[1,"\\n    "],[13],[1,"\\n  "]],[],false,["if"]]',moduleName:"/var/www/discourse/app/assets/javascripts/discourse/discourse/plugins/docker_manager/discourse/components/docker-manager/progress-bar.js",scope:()=>[s.default],isStrictMode:!0}),this))()}e.default=i})),define("discourse/plugins/docker_manager/discourse/components/docker-manager/upgrade-notice",["exports","@glimmer/component","@ember/routing","@ember/service","discourse-i18n","@ember/component","@ember/template-factory"],(function(e,t,r,s,n,a,i){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class o extends t.default{static#e=(()=>dt7948.g(this.prototype,"currentUser",[s.service]))()
#r=(()=>{dt7948.i(this,"currentUser")})()
static#t=(()=>(0,a.setComponentTemplate)((0,i.createTemplateFactory)({id:"aEyP0Qag",block:'[[[1,"\\n"],[41,[30,0,["currentUser","admin"]],[[[41,[51,[30,1,["upToDate"]]],[[[1,"        "],[10,0],[14,0,"upgrades-banner"],[12],[1,"\\n          "],[1,[28,[32,0],["admin.docker.outdated_notice"],null]],[1,"\\n\\n          "],[8,[32,1],null,[["@route"],["update"]],[["default"],[[[[1,"\\n            "],[1,[28,[32,0],["admin.docker.perform_update"],null]],[1,"\\n          "]],[]]]]],[1,"\\n        "],[13],[1,"\\n"]],[]],null]],[]],null],[1,"  "]],["@versionCheck"],false,["if","unless"]]',moduleName:"/var/www/discourse/app/assets/javascripts/discourse/discourse/plugins/docker_manager/discourse/components/docker-manager/upgrade-notice.js",scope:()=>[n.i18n,r.LinkTo],isStrictMode:!0}),this))()}e.default=o})),define("discourse/plugins/docker_manager/discourse/components/repo-status",["exports","@glimmer/component","@ember/object","@ember/service","discourse/components/d-button","discourse/helpers/d-icon","discourse/helpers/format-date","discourse-i18n","discourse/plugins/docker_manager/discourse/helpers/commit-url","discourse/plugins/docker_manager/discourse/helpers/new-commits","@ember/component","@ember/template-factory"],(function(e,t,r,s,n,a,i,o,d,u,l,c){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class p extends t.default{static#e=(()=>dt7948.g(this.prototype,"router",[s.service]))()
#s=(()=>{dt7948.i(this,"router")})()
static#t=(()=>dt7948.g(this.prototype,"upgradeStore",[s.service]))()
#n=(()=>{dt7948.i(this,"upgradeStore")})()
get upgradeDisabled(){return!this.args.upgradingRepo&&(!!this.upgradeStore.running||!this.args.managerRepo.upToDate&&this.args.managerRepo!==this.args.repo)}get upgradeButtonLabel(){return this.args.repo.upgrading?(0,o.i18n)("admin.docker.updating"):(0,o.i18n)("admin.docker.update_action")}upgrade(){this.router.transitionTo("update.show",this.args.repo)}static#a=(()=>dt7948.n(this.prototype,"upgrade",[r.action]))()
static#i=(()=>(0,l.setComponentTemplate)((0,c.createTemplateFactory)({id:"cVIlaUbr",block:'[[[1,"\\n    "],[10,"tr"],[15,0,[29,["d-admin-row__content repo ",[52,[30,1,["hasNewVersion"]],"has-update"]]]],[12],[1,"\\n      "],[10,"td"],[14,0,"d-admin-row__overview"],[12],[1,"\\n        "],[10,0],[14,0,"d-admin-row__overview-name"],[12],[1,"\\n          "],[1,[30,1,["nameTitleized"]]],[1,"\\n        "],[13],[1,"\\n\\n"],[41,[30,1,["author"]],[[[1,"          "],[10,0],[14,0,"d-admin-row__overview-author"],[12],[1,"\\n            "],[1,[30,1,["author"]]],[1,"\\n          "],[13],[1,"\\n"]],[]],null],[1,"\\n"],[41,[30,1,["plugin"]],[[[1,"          "],[10,0],[14,0,"d-admin-row__overview-about"],[12],[1,"\\n            "],[1,[30,1,["plugin","about"]]],[1,"\\n\\n"],[41,[30,1,["linkUrl"]],[[[1,"              "],[10,3],[15,6,[30,1,["linkUrl"]]],[14,"rel","noopener noreferrer"],[14,"target","_blank"],[12],[1,"\\n                "],[1,[28,[32,0],["admin.plugins.learn_more"],null]],[1,"\\n                "],[1,[28,[32,1],["up-right-from-square"],null]],[1,"\\n              "],[13],[1,"\\n"]],[]],null],[1,"          "],[13],[1,"\\n"]],[]],null],[1,"\\n"],[41,[30,1,["hasNewVersion"]],[[[1,"          "],[10,0],[14,0,"repo__new-version"],[12],[1,"\\n            "],[1,[28,[32,0],["admin.docker.new_version_available"],null]],[1,"\\n          "],[13],[1,"\\n"]],[]],null],[1,"      "],[13],[1,"\\n\\n      "],[10,"td"],[14,0,"d-admin-row__detail"],[12],[1,"\\n        "],[10,0],[14,0,"d-admin-row__mobile-label"],[12],[1,"\\n          "],[1,[28,[32,0],["admin.docker.repo.commit_hash"],null]],[1,"\\n        "],[13],[1,"\\n        "],[1,[28,[32,2],["current",[30,1,["version"]],[30,1,["prettyVersion"]],[30,1,["url"]]],null]],[1,"\\n      "],[13],[1,"\\n\\n      "],[10,"td"],[14,0,"d-admin-row__detail"],[12],[1,"\\n        "],[10,0],[14,0,"d-admin-row__mobile-label"],[12],[1,"\\n          "],[1,[28,[32,0],["admin.docker.repo.last_updated"],null]],[1,"\\n        "],[13],[1,"\\n        "],[1,[28,[32,3],[[30,1,["latest","date"]]],[["leaveAgo"],["true"]]]],[1,"\\n      "],[13],[1,"\\n\\n      "],[10,"td"],[14,0,"d-admin-row__detail"],[12],[1,"\\n        "],[10,0],[14,0,"d-admin-row__mobile-label"],[12],[1,"\\n          "],[1,[28,[32,0],["admin.docker.repo.latest_version"],null]],[1,"\\n        "],[13],[1,"\\n        "],[10,0],[14,0,"repo__latest-version"],[12],[1,"\\n          "],[10,0],[12],[1,"\\n            "],[1,[28,[32,2],["new",[30,1,["latest","version"]],[30,1,["prettyLatestVersion"]],[30,1,["url"]]],null]],[1,"\\n          "],[13],[1,"\\n          "],[10,0],[14,0,"new-commits"],[12],[1,"\\n            "],[1,[28,[32,4],[[30,1,["latest","commits_behind"]],[30,1,["version"]],[30,1,["latest","version"]],[30,1,["url"]]],null]],[1,"\\n          "],[13],[1,"\\n        "],[13],[1,"\\n      "],[13],[1,"\\n\\n      "],[10,"td"],[14,0,"d-admin-row__controls"],[12],[1,"\\n"],[41,[30,1,["checkingStatus"]],[[[1,"          "],[10,0],[14,0,"status-label --loading"],[12],[1,"\\n            "],[1,[28,[32,0],["admin.docker.checking"],null]],[1,"\\n          "],[13],[1,"\\n"]],[]],[[[41,[30,1,["upToDate"]],[[[1,"          "],[10,0],[14,"role","status"],[14,0,"status-label --success"],[12],[1,"\\n            "],[10,0],[14,0,"status-label-indicator"],[12],[1,"\\n            "],[13],[1,"\\n            "],[10,0],[14,0,"status-label-text"],[12],[1,"\\n              "],[1,[28,[32,0],["admin.docker.up_to_date"],null]],[1,"\\n            "],[13],[1,"\\n          "],[13],[1,"\\n"]],[]],[[[1,"          "],[8,[32,5],[[24,0,"upgrade-button"]],[["@action","@disabled","@translatedLabel"],[[30,0,["upgrade"]],[30,0,["upgradeDisabled"]],[30,0,["upgradeButtonLabel"]]]],null],[1,"\\n        "]],[]]]],[]]],[1,"      "],[13],[1,"\\n    "],[13],[1,"\\n  "]],["@repo"],false,["if"]]',moduleName:"/var/www/discourse/app/assets/javascripts/discourse/discourse/plugins/docker_manager/discourse/components/repo-status.js",scope:()=>[o.i18n,a.default,d.default,i.default,u.default,n.default],isStrictMode:!0}),this))()}e.default=p})),define("discourse/plugins/docker_manager/discourse/controllers/update-index",["exports","@glimmer/tracking","@ember/controller","@ember/object","@ember/service","discourse/plugins/docker_manager/discourse/models/repo"],(function(e,t,r,s,n,a){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class i extends r.default{static#e=(()=>dt7948.g(this.prototype,"router",[n.service]))()
#s=(()=>{dt7948.i(this,"router")})()
get managerRepo(){return this.model.find((e=>"docker_manager"===e.id))}static#t=(()=>dt7948.n(this.prototype,"managerRepo",[t.cached]))()
get outdated(){return a.needsImageUpgrade}get upgradeAllButtonDisabled(){return!this.managerRepo.upToDate||this.allUpToDate}get allUpToDate(){return this.model.every((e=>e.upToDate))}upgradeAllButton(){this.router.transitionTo("update.show","all")}static#a=(()=>dt7948.n(this.prototype,"upgradeAllButton",[s.action]))()}e.default=i})),define("discourse/plugins/docker_manager/discourse/controllers/update-show",["exports","@ember/controller","@ember/object","@ember/service","discourse/lib/helpers","discourse-i18n","discourse/plugins/docker_manager/discourse/models/repo"],(function(e,t,r,s,n,a,i){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class o extends t.default{static#e=(()=>dt7948.g(this.prototype,"dialog",[s.service]))()
#o=(()=>{dt7948.i(this,"dialog")})()
static#t=(()=>dt7948.g(this.prototype,"upgradeStore",[s.service]))()
#n=(()=>{dt7948.i(this,"upgradeStore")})()
get complete(){return"complete"===this.upgradeStore.upgradeStatus}get failed(){return"failed"===this.upgradeStore.upgradeStatus}get multiUpgrade(){return this.model.length>1}get title(){return this.multiUpgrade?(0,a.i18n)("admin.docker.update_everything"):(0,a.i18n)("admin.docker.update_repo",{name:this.model.name})}get isUpToDate(){return(0,n.makeArray)(this.model).every((e=>e.upToDate))}get upgrading(){return(0,n.makeArray)(this.model).some((e=>e.upgrading))}start(){if(this.upgradeStore.reset(),this.multiUpgrade){for(const e of this.model)e.upToDate||(e.upgrading=!0)
return i.default.upgradeAll()}if(!this.model.upgrading)return this.model.startUpgrade()}static#a=(()=>dt7948.n(this.prototype,"start",[r.action]))()
resetUpgrade(){this.dialog.confirm({message:(0,a.i18n)("admin.docker.reset_warning"),didConfirm:async()=>{if(this.multiUpgrade)try{await i.default.resetAll(this.model.filter((e=>!e.upToDate)))}finally{this.upgradeStore.reset()
for(const e of this.model)e.upgrading=!1}else await this.model.resetUpgrade(),this.upgradeStore.reset()}})}static#i=(()=>dt7948.n(this.prototype,"resetUpgrade",[r.action]))()}e.default=o})),define("discourse/plugins/docker_manager/discourse/controllers/update",["exports","@glimmer/tracking","@ember/controller","@ember/object","@ember-compat/tracked-built-ins"],(function(e,t,r,s,n){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class a extends r.default{static#e=(()=>dt7948.g(this.prototype,"bannerDismissed",[t.tracked],(function(){return!1})))()
#d=(()=>{dt7948.i(this,"bannerDismissed")})()
banner=(()=>new n.TrackedArray)()
get showBanner(){return!this.bannerDismissed&&this.banner?.length>0}appendBannerHtml(e){this.banner.includes(e)||this.banner.push(e)}dismiss(){this.bannerDismissed=!0}static#t=(()=>dt7948.n(this.prototype,"dismiss",[s.action]))()}e.default=a})),define("discourse/plugins/docker_manager/discourse/docker-manager-route-map",["exports"],(function(e){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
e.default={resource:"admin",map(){this.route("update",{resetNamespace:!0},(function(){this.route("processes"),this.route("show",{path:"/:id"})}))}}})),define("discourse/plugins/docker_manager/discourse/helpers/commit-url",["exports","@ember/template"],(function(e,t){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=function(e,r,s,n){if(!s)return""
if(!n)return s
const a=n.substr(0,n.search(/(\.git)?$/))
return(0,t.htmlSafe)(`<a class='${e} commit-hash' title='${r}' href='${a}/commit/${r}'>${s}</a>`)}})),define("discourse/plugins/docker_manager/discourse/helpers/new-commits",["exports","@ember/template","discourse-i18n"],(function(e,t,r){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=function(e,s,n,a){if(!e)return""
if(0===parseInt(e,10))return""
const i=(0,r.i18n)("admin.docker.commits",{count:e})
if(!a)return i
const o=a.substr(0,a.search(/(\.git)?$/))
return(0,t.htmlSafe)(`<a href='${o}/compare/${s}...${n}'>${i}</a>`)}})),define("discourse/plugins/docker_manager/discourse/initializers/admin-sidebar",["exports","discourse/lib/plugin-api"],(function(e,t){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
e.default={name:"docker-manager-admin-sidebar",initialize(){(0,t.withPluginApi)("1.24.0",(e=>{e.addAdminSidebarSectionLink("root",{name:"admin_upgrade",route:"update.index",label:"admin.docker.update_tab",icon:"rocket"})}))}}})),define("discourse/plugins/docker_manager/discourse/models/process-list",["exports","@glimmer/tracking","discourse/lib/ajax"],(function(e,t,r){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class s{static#e=(()=>dt7948.g(this.prototype,"output",[t.tracked],(function(){return null})))()
#u=(()=>{dt7948.i(this,"output")})()
async refresh(){const e=await(0,r.ajax)("/admin/docker/ps",{dataType:"text"})
this.output=e}}e.default=s})),define("discourse/plugins/docker_manager/discourse/models/repo",["exports","@glimmer/tracking","@ember/string","@ember-compat/tracked-built-ins","discourse/lib/ajax","admin/models/admin-plugin"],(function(e,t,r,s,n,a){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.needsImageUpgrade=e.default=void 0
let i=[],o=e.needsImageUpgrade=!1
function d(e){return e.map((e=>e.version)).join(", ")}class u{static create(){return new u(...arguments)}static async findAll(){if(i.length)return i
const t=await(0,n.ajax)("/admin/docker/repos")
return i=t.repos.map((e=>new u(e))),e.needsImageUpgrade=o=t.upgrade_required,i}static async findUpgrading(){return(await u.findAll()).findBy("upgrading",!0)}static async find(e){return(await u.findAll()).findBy("id",e)}static upgradeAll(){return(0,n.ajax)("/admin/docker/update",{dataType:"text",type:"POST",data:{path:"all"}})}static resetAll(e){return(0,n.ajax)("/admin/docker/update",{dataType:"text",type:"DELETE",data:{path:"all",version:d(e)}})}static async findLatestAll(){return(await(0,n.ajax)("/admin/docker/latest",{dataType:"json",type:"GET",data:{path:"all"}})).repos}static async findAllProgress(e){return(await(0,n.ajax)("/admin/docker/progress",{dataType:"json",type:"GET",data:{path:"all",version:d(e)}})).progress}static#e=(()=>dt7948.g(this.prototype,"unloaded",[t.tracked],(function(){return!0})))()
#l=(()=>{dt7948.i(this,"unloaded")})()
static#t=(()=>dt7948.g(this.prototype,"checking",[t.tracked],(function(){return!1})))()
#c=(()=>{dt7948.i(this,"checking")})()
static#a=(()=>dt7948.g(this.prototype,"lastCheckedAt",[t.tracked],(function(){return null})))()
#p=(()=>{dt7948.i(this,"lastCheckedAt")})()
static#i=(()=>dt7948.g(this.prototype,"latest",[t.tracked],(function(){return new s.TrackedObject({})})))()
#g=(()=>{dt7948.i(this,"latest")})()
static#m=(()=>dt7948.g(this.prototype,"plugin",[t.tracked],(function(){return null})))()
#h=(()=>{dt7948.i(this,"plugin")})()
static#f=(()=>dt7948.g(this.prototype,"name",[t.tracked],(function(){return null})))()
#_=(()=>{dt7948.i(this,"name")})()
static#b=(()=>dt7948.g(this.prototype,"path",[t.tracked],(function(){return null})))()
#k=(()=>{dt7948.i(this,"path")})()
static#v=(()=>dt7948.g(this.prototype,"branch",[t.tracked],(function(){return null})))()
#y=(()=>{dt7948.i(this,"branch")})()
static#w=(()=>dt7948.g(this.prototype,"official",[t.tracked],(function(){return!1})))()
#x=(()=>{dt7948.i(this,"official")})()
static#T=(()=>dt7948.g(this.prototype,"fork",[t.tracked],(function(){return!1})))()
#j=(()=>{dt7948.i(this,"fork")})()
static#S=(()=>dt7948.g(this.prototype,"id",[t.tracked],(function(){return null})))()
#O=(()=>{dt7948.i(this,"id")})()
static#P=(()=>dt7948.g(this.prototype,"version",[t.tracked],(function(){return null})))()
#M=(()=>{dt7948.i(this,"version")})()
static#U=(()=>dt7948.g(this.prototype,"pretty_version",[t.tracked],(function(){return null})))()
#A=(()=>{dt7948.i(this,"pretty_version")})()
static#D=(()=>dt7948.g(this.prototype,"url",[t.tracked],(function(){return null})))()
#B=(()=>{dt7948.i(this,"url")})()
static#C=(()=>dt7948.g(this.prototype,"upgrading",[t.tracked],(function(){return!1})))()
#R=(()=>{dt7948.i(this,"upgrading")})()
constructor(){let e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{}
if(e.latest)for(const[t,r]of Object.entries(e.latest))this.latest[t]=r
e.plugin&&(this.plugin=a.default.create(e.plugin))
for(const[t,r]of Object.entries(e))["latest","plugin"].includes(t)||(this[t]=r)}get nameTitleized(){return this.plugin?this.plugin.nameTitleized:(0,r.capitalize)(this.name)}static#L=(()=>dt7948.n(this.prototype,"nameTitleized",[t.cached]))()
get linkUrl(){return this.plugin?this.plugin.linkUrl:this.url}get author(){return this.plugin?this.plugin.author:null}get checkingStatus(){return this.unloaded||this.checking}get upToDate(){return!this.upgrading&&this.version===this.latest?.version}get hasNewVersion(){return!this.checkingStatus&&!this.upToDate}get prettyVersion(){return this.pretty_version||this.version?.substring(0,8)}get prettyLatestVersion(){return this.latest?.pretty_version||this.latest?.version?.substring(0,8)}get shouldCheck(){if(null===this.version)return!1
if(this.checking)return!1
if(this.lastCheckedAt){return(new Date).getTime()-this.lastCheckedAt>6e4}return!0}repoAjax(e){let t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{}
return t.data={path:this.path,version:this.version,branch:this.branch},(0,n.ajax)(e,t)}async findLatest(){if(!this.shouldCheck)return void(this.unloaded=!1)
this.checking=!0
const e=await this.repoAjax("/admin/docker/latest")
this.unloaded=!1,this.checking=!1,this.lastCheckedAt=(new Date).getTime()
for(const[t,r]of Object.entries(e.latest))this.latest[t]=r}async findProgress(){return(await this.repoAjax("/admin/docker/progress")).progress}async resetUpgrade(){await this.repoAjax("/admin/docker/update",{dataType:"text",type:"DELETE"}),this.upgrading=!1}async startUpgrade(){this.upgrading=!0
try{await this.repoAjax("/admin/docker/update",{dataType:"text",type:"POST"})}catch{this.upgrading=!1}}}e.default=u})),define("discourse/plugins/docker_manager/discourse/routes/update-index",["exports","@ember/routing/route"],(function(e,t){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class r extends t.default{model(){return this.modelFor("update")}async loadRepos(e){for(const t of e)await t.findLatest()}setupController(e,t){super.setupController(...arguments),this.loadRepos(t)}}e.default=r})),define("discourse/plugins/docker_manager/discourse/routes/update-processes",["exports","@ember/routing/route","@ember/runloop","discourse/lib/decorators","discourse/lib/later","discourse/plugins/docker_manager/discourse/models/process-list"],(function(e,t,r,s,n,a){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class i extends t.default{processes=null
refreshTimer=null
autoRefresh=!1
model(){return this.processes=new a.default,this.autoRefresh=!0,this.refresh(),this.processes}deactivate(){this.autoRefresh=!1}async refresh(){this.autoRefresh?(await this.processes.refresh(),this.refreshTimer=(0,n.default)(this.refresh,5e3)):(0,r.cancel)(this.refreshTimer)}static#e=(()=>dt7948.n(this.prototype,"refresh",[s.bind]))()}e.default=i})),define("discourse/plugins/docker_manager/discourse/routes/update-show",["exports","@ember/routing/route","@ember/service","discourse/plugins/docker_manager/discourse/models/repo"],(function(e,t,r,s){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class n extends t.default{static#e=(()=>dt7948.g(this.prototype,"upgradeStore",[r.service]))()
#n=(()=>{dt7948.i(this,"upgradeStore")})()
static#t=(()=>dt7948.g(this.prototype,"router",[r.service]))()
#s=(()=>{dt7948.i(this,"router")})()
model(e){return"all"===e.id?s.default.findAll():s.default.find(e.id)}async afterModel(e){if(!e)return void this.router.replaceWith("/404")
if(Array.isArray(e)){const t=await s.default.findLatestAll()
for(const s of t){const t=e.find((e=>e.path===s.path))
if(!t)return
delete s.path
for(const[e,r]of Object.entries(s))t.latest[e]=r}const r=await s.default.findAllProgress(e.filter((e=>!e.upToDate)))
return void this.upgradeStore.reset({consoleOutput:r.logs,progressPercentage:r.percentage,upgradeStatus:r.status,repos:t})}await s.default.findUpgrading(),await e.findLatest()
const t=await e.findProgress()
this.upgradeStore.reset({consoleOutput:t.logs,progressPercentage:t.percentage,upgradeStatus:t.status,repos:[e.id]})}}e.default=n})),define("discourse/plugins/docker_manager/discourse/routes/update",["exports","@ember/routing/route","@ember/service","discourse/lib/decorators","discourse-i18n","discourse/plugins/docker_manager/discourse/models/repo"],(function(e,t,r,s,n,a){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class i extends t.default{static#e=(()=>dt7948.g(this.prototype,"messageBus",[r.service]))()
#N=(()=>{dt7948.i(this,"messageBus")})()
static#t=(()=>dt7948.g(this.prototype,"upgradeStore",[r.service]))()
#n=(()=>{dt7948.i(this,"upgradeStore")})()
model(){return a.default.findAll()}activate(){this.messageBus.subscribe("/docker/update",this.onUpgradeMessage)}deactivate(){this.messageBus.unsubscribe("/docker/update",this.onUpgradeMessage)}setupController(e,t){const r=t.find((e=>"discourse"===e.id))
"origin/main"===r?.branch&&e.appendBannerHtml((0,n.i18n)("admin.docker.main_branch_warning",{url:"https://meta.discourse.org/t/17014"}))}onUpgradeMessage(e){switch(e.type){case"log":this.upgradeStore.consoleOutput=this.upgradeStore.consoleOutput+e.value+"\n"
break
case"percent":this.upgradeStore.progressPercentage=e.value
break
case"status":this.upgradeStore.upgradeStatus=e.value
const t=this.modelFor("update")
if("complete"===e.value){for(const e of t)e.upgrading&&(e.version=e.latest?.version)
this.session.requiresRefresh=!0}if("complete"===e.value||"failed"===e.value)for(const e of t)e.upgrading=!1}}static#a=(()=>dt7948.n(this.prototype,"onUpgradeMessage",[s.bind]))()}e.default=i})),define("discourse/plugins/docker_manager/discourse/services/upgrade-store",["exports","@glimmer/tracking","@ember/service"],(function(e,t,r){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
class s extends r.default{static#e=(()=>dt7948.g(this.prototype,"consoleOutput",[t.tracked],(function(){return""})))()
#F=(()=>{dt7948.i(this,"consoleOutput")})()
static#t=(()=>dt7948.g(this.prototype,"progressPercentage",[t.tracked],(function(){return 0})))()
#$=(()=>{dt7948.i(this,"progressPercentage")})()
static#a=(()=>dt7948.g(this.prototype,"upgradeStatus",[t.tracked],(function(){return null})))()
#V=(()=>{dt7948.i(this,"upgradeStatus")})()
static#i=(()=>dt7948.g(this.prototype,"repos",[t.tracked],(function(){return[]})))()
#E=(()=>{dt7948.i(this,"repos")})()
get running(){return"running"===this.upgradeStatus}reset(){let{consoleOutput:e,progressPercentage:t,upgradeStatus:r,repos:s}=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{}
this.consoleOutput=e??"",this.progressPercentage=t??0,this.upgradeStatus=r??null,this.repos=s??[]}}e.default=s})),define("discourse/plugins/docker_manager/discourse/templates/connectors/admin-menu/upgrade-link",["exports","@ember/template-factory"],(function(e,t){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
e.default=(0,t.createTemplateFactory)({id:"NugvYRqo",block:'[[[41,[30,0,["currentUser","admin"]],[[[1,"  "],[8,[39,1],null,[["@route","@label"],["update","admin.docker.update_tab"]],null],[1,"\\n"]],[]],null]],[],false,["if","nav-item"]]',moduleName:"discourse/plugins/docker_manager/discourse/templates/connectors/admin-menu/upgrade-link.hbs",isStrictMode:!1})})),define("discourse/plugins/docker_manager/discourse/templates/connectors/admin-upgrade-header/upgrade-header",["exports","@ember/template-factory"],(function(e,t){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
e.default=(0,t.createTemplateFactory)({id:"8s3istjp",block:'[[[8,[39,0],null,[["@versionCheck"],[[30,0,["versionCheck"]]]],null]],[],false,["docker-manager/upgrade-notice"]]',moduleName:"discourse/plugins/docker_manager/discourse/templates/connectors/admin-upgrade-header/upgrade-header.hbs",isStrictMode:!1})})),define("discourse/plugins/docker_manager/discourse/templates/update-index",["exports","@ember/template-factory"],(function(e,t){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
e.default=(0,t.createTemplateFactory)({id:"LNovkYVK",block:'[[[10,0],[14,0,"updates-heading"],[12],[1,"\\n"],[41,[51,[30,0,["outdated"]]],[[[1,"    "],[8,[39,2],[[16,"disabled",[30,0,["upgradeAllButtonDisabled"]]],[24,1,"upgrade-all"],[24,0,"btn btn-primary"],[24,4,"button"],[4,[38,3],["click",[30,0,["upgradeAllButton"]]],null]],null,[["default"],[[[[1,"\\n"],[41,[30,0,["allUpToDate"]],[[[1,"        "],[1,[28,[35,5],["admin.docker.all_up_to_date"],null]],[1,"\\n"]],[]],[[[1,"        "],[1,[28,[35,5],["admin.docker.update_all"],null]],[1,"\\n"]],[]]],[1,"    "]],[]]]]],[1,"\\n"]],[]],null],[13],[1,"\\n\\n"],[41,[30,0,["outdated"]],[[[1,"  "],[10,"h2"],[12],[1,[28,[35,5],["admin.docker.outdated_image_header"],null]],[13],[1,"\\n  "],[10,2],[12],[1,[28,[35,5],["admin.docker.outdated_image_info"],null]],[13],[1,"\\n\\n"],[1,"  "],[10,"pre"],[12],[1,"    cd /var/discourse\\n    ./launcher rebuild app\\n  "],[13],[1,"\\n  "],[10,2],[12],[1,"\\n    "],[10,3],[14,6,"https://meta.discourse.org/t/how-do-i-update-my-docker-image-to-latest/23325"],[12],[1,"\\n      "],[1,[28,[35,5],["admin.docker.outdated_image_link"],null]],[1,"\\n    "],[13],[1,"\\n  "],[13],[1,"\\n"]],[]],[[[1,"\\n  "],[10,"table"],[14,0,"d-admin-table"],[14,1,"repos"],[12],[1,"\\n    "],[10,"thead"],[12],[1,"\\n      "],[10,"th"],[12],[1,[28,[35,5],["admin.docker.repo.name"],null]],[13],[1,"\\n      "],[10,"th"],[12],[1,[28,[35,5],["admin.docker.repo.commit_hash"],null]],[13],[1,"\\n      "],[10,"th"],[12],[1,[28,[35,5],["admin.docker.repo.last_updated"],null]],[13],[1,"\\n      "],[10,"th"],[12],[1,[28,[35,5],["admin.docker.repo.latest_version"],null]],[13],[1,"\\n      "],[10,"th"],[12],[1,[28,[35,5],["admin.docker.repo.status"],null]],[13],[1,"\\n    "],[13],[1,"\\n    "],[10,"tbody"],[12],[1,"\\n"],[42,[28,[37,15],[[28,[37,15],[[30,0,["model"]]],null]],null],null,[[[1,"        "],[8,[39,16],null,[["@repo","@upgradingRepo","@managerRepo"],[[30,1],[30,1,["upgrading"]],[30,0,["managerRepo"]]]],null],[1,"\\n"]],[1]],null],[1,"    "],[13],[1,"\\n  "],[13],[1,"\\n"]],[]]]],["repo"],false,["div","unless","d-button","on","if","i18n","h2","p","pre","a","table","thead","th","tbody","each","-track-array","repo-status"]]',moduleName:"discourse/plugins/docker_manager/discourse/templates/update-index.hbs",isStrictMode:!1})})),define("discourse/plugins/docker_manager/discourse/templates/update-processes",["exports","@ember/template-factory"],(function(e,t){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
e.default=(0,t.createTemplateFactory)({id:"Y6yF+lGW",block:'[[[8,[39,0],null,[["@output"],[[30,0,["model","output"]]]],null]],[],false,["docker-manager/console"]]',moduleName:"discourse/plugins/docker_manager/discourse/templates/update-processes.hbs",isStrictMode:!1})})),define("discourse/plugins/docker_manager/discourse/templates/update-show",["exports","@ember/template-factory"],(function(e,t){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
e.default=(0,t.createTemplateFactory)({id:"gMBYcIgd",block:'[[[10,"h2"],[12],[1,[30,0,["title"]]],[13],[1,"\\n\\n"],[8,[39,1],null,[["@percent"],[[30,0,["upgradeStore","progressPercentage"]]]],null],[1,"\\n\\n"],[41,[30,0,["complete"]],[[[1,"  "],[10,2],[12],[1,[28,[35,4],["admin.docker.update_successful"],null]],[13],[1,"\\n"]],[]],[[[41,[30,0,["failed"]],[[[1,"  "],[10,2],[12],[1,[28,[35,4],["admin.docker.update_error"],null]],[13],[1,"\\n"]],[]],null]],[]]],[1,"\\n"],[41,[30,0,["isUpToDate"]],[[[41,[30,0,["multiUpgrade"]],[[[1,"    "],[10,2],[12],[1,[28,[35,4],["admin.docker.everything_up_to_date"],null]],[13],[1,"\\n"]],[]],[[[1,"    "],[10,2],[12],[1,[28,[35,4],["admin.docker.repo_newest_version"],[["name"],[[30,0,["model","name"]]]]]],[13],[1,"\\n"]],[]]]],[]],[[[1,"  "],[10,0],[14,0,"upgrade-actions"],[12],[1,"\\n    "],[11,"button"],[16,"disabled",[30,0,["upgrading"]]],[24,0,"btn start-upgrade"],[24,4,"button"],[4,[38,7],["click",[30,0,["start"]]],null],[12],[1,"\\n"],[41,[30,0,["upgrading"]],[[[1,"        "],[1,[28,[35,4],["admin.docker.updating"],null]],[1,"\\n"]],[]],[[[1,"        "],[1,[28,[35,4],["admin.docker.start_updating"],null]],[1,"\\n"]],[]]],[1,"    "],[13],[1,"\\n\\n"],[41,[30,0,["upgrading"]],[[[1,"      "],[11,"button"],[24,0,"btn unlock"],[24,4,"button"],[4,[38,7],["click",[30,0,["resetUpgrade"]]],null],[12],[1,"\\n        "],[1,[28,[35,4],["admin.docker.reset_update"],null]],[1,"\\n      "],[13],[1,"\\n"]],[]],null],[1,"  "],[13],[1,"\\n"]],[]]],[1,"\\n"],[8,[39,8],null,[["@output","@followOutput"],[[30,0,["upgradeStore","consoleOutput"]],true]],null]],[],false,["h2","docker-manager/progress-bar","if","p","i18n","div","button","on","docker-manager/console"]]',moduleName:"discourse/plugins/docker_manager/discourse/templates/update-show.hbs",isStrictMode:!1})})),define("discourse/plugins/docker_manager/discourse/templates/update",["exports","@ember/template-factory"],(function(e,t){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
e.default=(0,t.createTemplateFactory)({id:"uVwdBKPK",block:'[[[8,[39,0],null,[["@titleLabel","@descriptionLabel","@shouldDisplay"],[[28,[37,1],["admin.docker.update_title"],null],[28,[37,1],["admin.docker.update_description"],null],true]],[["breadcrumbs","tabs"],[[[[1,"\\n    "],[8,[39,3],null,[["@path","@label"],["/admin",[28,[37,1],["admin_title"],null]]],null],[1,"\\n    "],[8,[39,3],null,[["@path","@label"],["/admin/update",[28,[37,1],["admin.docker.update_title"],null]]],null],[1,"\\n  "]],[]],[[[1,"\\n    "],[8,[39,5],null,[["@route","@label"],["update.index","admin.docker.navigation.versions"]],null],[1,"\\n    "],[8,[39,5],null,[["@route","@label"],["update.processes","admin.docker.navigation.processes"]],null],[1,"\\n  "]],[]]]]],[1,"\\n\\n"],[10,0],[14,0,"docker-manager admin-container"],[12],[1,"\\n"],[41,[30,0,["showBanner"]],[[[1,"    "],[10,0],[14,1,"banner"],[12],[1,"\\n      "],[10,0],[14,1,"banner-content"],[12],[1,"\\n        "],[10,0],[14,0,"floated-buttons"],[12],[1,"\\n          "],[8,[39,8],[[24,0,"btn btn-flat close"]],[["@icon","@action","@title"],["xmark",[30,0,["dismiss"]],"banner.close"]],null],[1,"\\n        "],[13],[1,"\\n\\n"],[42,[28,[37,10],[[28,[37,10],[[30,0,["banner"]]],null]],null],null,[[[1,"          "],[10,2],[12],[1,[28,[35,12],[[30,1]],null]],[13],[1,"\\n"]],[1]],null],[1,"      "],[13],[1,"\\n    "],[13],[1,"\\n"]],[]],null],[1,"\\n  "],[46,[28,[37,14],null,null],null,null,null],[1,"\\n"],[13]],["row"],false,["d-page-header","i18n",":breadcrumbs","d-breadcrumbs-item",":tabs","nav-item","div","if","d-button","each","-track-array","p","html-safe","component","-outlet"]]',moduleName:"discourse/plugins/docker_manager/discourse/templates/update.hbs",isStrictMode:!1})}))

//# sourceMappingURL=docker_manager_admin-2c839c98bdfe4a00dda4cdeaba49fe771194f364e9a6ec796c40f712fcc9af0f.map
//!

;
