import{loadData as M}from"./data-BCva9ti9.js";const E={"getting-started":`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 00-2.91-.09z"/>
    <path d="M12 15l-3-3a22 22 0 012-3.95A12.88 12.88 0 0122 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 01-4 2z"/>
    <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/>
    <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>
  </svg>`,"chat-loop":`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
  </svg>`,configuration:`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="3"/>
    <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/>
  </svg>`,extensions:`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M13.5 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8.5L13.5 2z"/>
    <path d="M14 2v6h6"/>
    <path d="M8 13h3"/>
    <path d="M8 17h6"/>
    <path d="M9.5 9.5L11 11l2.5-2.5"/>
  </svg>`,"session-lifecycle":`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <polyline points="12 6 12 12 16 14"/>
  </svg>`,"under-the-hood":`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/>
  </svg>`,telemetry:`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M18 20V10"/>
    <path d="M12 20V4"/>
    <path d="M6 20v-6"/>
  </svg>`},N=`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10"/>
  <line x1="12" y1="8" x2="12" y2="12"/>
  <line x1="12" y1="16" x2="12.01" y2="16"/>
</svg>`;async function j(p,g){let u;try{u=await M("workflows.json")}catch(e){return console.warn("workflow-nav: could not load workflows.json",e),p.innerHTML='<div class="workflow-nav"><p style="padding:16px;color:var(--text-muted);font-size:0.82rem;">No workflows available.</p></div>',{setActive(){}}}const v=u.categories||[],C=u.workflows||[];for(const e of v)e.workflows||(e.workflows=C.filter(t=>t.category===e.id));const d=document.createElement("div");d.className="workflow-nav";const w=document.createElement("div");w.className="nav-search";const a=document.createElement("input");a.type="text",a.className="search-input",a.placeholder="Search workflows…",w.appendChild(a),d.appendChild(w);const h=document.createElement("div");h.className="nav-categories";const m=new Map,y=new Map;for(const e of v){const t=document.createElement("div");t.className="nav-category";const n=document.createElement("div");n.className="nav-category-header";const s=document.createElement("span");s.className="nav-category-icon",s.innerHTML=E[e.id]||N;const o=document.createElement("span");o.className="nav-category-title",o.textContent=e.title;const i=document.createElement("span");i.className="nav-category-count",i.textContent=(e.workflows||[]).length,n.append(s,o,i),t.appendChild(n);const l=document.createElement("ul");l.className="nav-category-items";for(const r of e.workflows||[]){const c=document.createElement("li");c.className="nav-item",c.setAttribute("data-id",r.id);const f=document.createElement("span");f.className="nav-item-title",f.textContent=r.title;const k=document.createElement("span");k.className="nav-item-subtitle",k.textContent=r.subtitle||"",c.append(f,k),c.addEventListener("click",()=>{x(r.id),g&&g(r.id)}),l.appendChild(c),m.set(r.id,c)}t.appendChild(l),h.appendChild(t),y.set(e.id,t)}d.appendChild(h),p.innerHTML="",p.appendChild(d),a.addEventListener("input",()=>{const e=a.value.trim().toLowerCase();for(const t of v){const n=y.get(t.id);let s=0;for(const o of t.workflows||[]){const i=m.get(o.id),l=!e||o.title.toLowerCase().includes(e)||o.subtitle&&o.subtitle.toLowerCase().includes(e);i.style.display=l?"":"none",l&&s++}n.style.display=s>0?"":"none"}});function x(e){for(const[t,n]of m)n.classList.toggle("active",t===e)}return{setActive:x}}export{j as createWorkflowNav};
