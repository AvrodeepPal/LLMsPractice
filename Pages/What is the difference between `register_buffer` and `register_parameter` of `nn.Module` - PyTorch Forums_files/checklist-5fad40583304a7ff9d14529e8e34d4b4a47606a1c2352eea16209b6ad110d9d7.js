define("discourse/plugins/checklist/discourse/initializers/checklist",["exports","discourse/lib/ajax","discourse/lib/ajax-error","discourse/lib/icon-library","discourse/lib/plugin-api","discourse-i18n","discourse/plugins/checklist/lib/rich-editor-extension"],(function(e,t,n,c,s,i,a){"use strict"
function r(e){return 3===e.nodeType&&e.nodeValue.match(/^\s*$/)}function o(e){e.forEach((e=>{let t=e.parentElement
"P"===t.nodeName&&t.parentElement.firstElementChild===t&&(t=t.parentElement),"LI"!==t.nodeName||"UL"!==t.parentElement.nodeName||function(e){let t=e.previousSibling
for(;t;){if(!r(t))return!0
t=t.previousSibling}return!1}(e)||(t.classList.add("has-checkbox"),e.classList.add("list-item-checkbox"),e.nextSibling||e.insertAdjacentHTML("afterend","&#8203;"))}))}function l(e,s){const a=[...e.getElementsByClassName("chcklst-box")]
o(a)
const r=s?.widget,l=s?.getModel()
l?.can_edit&&a.forEach(((e,s)=>{e.onclick=async e=>{const o=e.currentTarget,d=o.classList
if(d.contains("permanent")||d.contains("readonly"))return
const u=d.contains("checked")?"[ ]":"[x]",h=document.createElement("template")
h.innerHTML=(0,c.iconHTML)("spinner",{class:"fa-spin list-item-checkbox"}),o.insertAdjacentElement("afterend",h.content.firstChild),o.classList.add("hidden"),a.forEach((e=>e.classList.add("readonly")))
try{const e=await(0,t.ajax)(`/posts/${l.id}`),n=[];[/`[^`\n]*\n?[^`\n]*`/gm,/^```[^]*?^```/gm,/\[code\][^]*?\[\/code\]/gm,/_(?=\S).*?\S_/gm,/~~(?=\S).*?\S~~/gm].forEach((t=>{let c
for(;null!=(c=t.exec(e.raw));)n.push([c.index,c.index+c[0].length])})),[/([^\[\n]|^)\*\S.+?\S\*(?=[^\]\n]|$)/gm].forEach((t=>{let c
for(;null!=(c=t.exec(e.raw));)n.push([c.index+1,c.index+c[0].length])}))
let c=-1,a=!1
const o=e.raw.replace(/\[( |x)?\]/gi,((t,i,r)=>a||r>0&&"!"===e.raw[r-1]?t:(c+=n.every((e=>e[0]>=r+t.length||r>e[1])),c===s?(a=!0,u):t)))
await l.save({raw:o,edit_reason:(0,i.i18n)("checklist.edit_reason")}),r.attrs.isSaving=!1,r.scheduleRerender()}catch(f){(0,n.popupAjaxError)(f)}finally{a.forEach((e=>e.classList.remove("readonly"))),o.classList.remove("hidden"),o.parentElement.querySelector(".fa-spin").remove()}}}))}Object.defineProperty(e,"__esModule",{value:!0}),e.checklistSyntax=l,e.default=void 0
e.default={name:"checklist",initialize(){(0,s.withPluginApi)("0.1",(e=>function(e){e.container.lookup("service:site-settings").checklist_enabled&&(e.decorateCookedElement(l),e.registerRichEditorExtension(a.default))}(e)))}}})),define("discourse/plugins/checklist/lib/discourse-markdown/checklist",["exports"],(function(e){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.setup=function(e){e.registerOptions(((e,t)=>{e.features.checklist=!!t.checklist_enabled})),e.allowList(["span.chcklst-stroked","span.chcklst-box fa fa-square-o fa-fw","span.chcklst-box checked fa fa-square-check-o fa-fw","span.chcklst-box checked permanent fa fa-square-check fa-fw"]),e.registerPlugin((e=>e.core.ruler.push("checklist",s)))}
const t=/\[( |x)?\]/gi
function n(e,t,n,c){const s=function(e){switch(e){case"x":return"checked fa fa-square-check-o fa-fw"
case"X":return"checked permanent fa fa-square-check fa-fw"
default:return"fa fa-square-o fa-fw"}}(n[1]),i=new c.Token("check_open","span",1)
i.attrs=[["class",`chcklst-box ${s}`]],e.push(i)
const a=new c.Token("check_close","span",-1)
e.push(a)}function c(e,c){let s,i=null,a=0
for(;s=t.exec(e);){if(s.index>a){i=i||[]
const t=new c.Token("text","",0)
t.content=e.slice(a,s.index),i.push(t)}a=s.index+s[0].length,i=i||[],n(i,0,s,c)}if(i&&a<e.length){const t=new c.Token("text","",0)
t.content=e.slice(a),i.push(t)}return i}function s(e){let t,n,s,i,a,r=e.tokens,o=0
for(n=0,s=r.length;n<s;n++)if("inline"===r[n].type)for(i=r[n].children,t=i.length-1;t>=0;t--)if(a=i[t],o+=a.nesting,"text"===a.type&&0===o){const s=c(a.content,e)
s&&(r[n].children=i=e.md.utils.arrayReplaceAt(i,t,s))}}})),define("discourse/plugins/checklist/lib/rich-editor-extension",["exports"],(function(e){"use strict"
Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0
const t={nodeSpec:{check:{attrs:{checked:{default:!1}},inline:!0,group:"inline",draggable:!0,selectable:!1,toDOM:e=>["span",{class:e.attrs.checked?"chcklst-box checked fa fa-square-check-o fa-fw":"chcklst-box fa fa-square-o fa-fw"}],parseDOM:[{tag:"span.chcklst-box",getAttrs:e=>({checked:c(e.className)})}]}},inputRules:[{match:/(^|\s)\[(x? ?)]$/,handler:(e,t,n,c)=>e.tr.replaceWith(n+t[1].length,c,e.schema.nodes.check.create({checked:"x"===t[2]})),options:{undoable:!1}}],parse:{check_open:{node:"check",getAttrs:e=>({checked:c(e.attrGet("class"))})},check_close:{noCloseToken:!0,ignore:!0}},serializeNode:{check:(e,t)=>{e.write(t.attrs.checked?"[x]":"[ ]")}}},n=/\bchecked\b/
function c(e){return n.test(e)}e.default=t}))

//# sourceMappingURL=checklist-f6ba1329af10fa17503a73d046131f9944b35327e339af84cade6b876152f5ef.map
//!

;
