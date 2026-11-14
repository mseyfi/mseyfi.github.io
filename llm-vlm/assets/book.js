/* LLM & VLM course: shared runtime */
(function(){
"use strict";
const B=window.Book=window.Book||{};
B.$=(s,r)=>(r||document).querySelector(s);B.$$=(s,r)=>Array.from((r||document).querySelectorAll(s));
B.reduced=window.matchMedia&&matchMedia("(prefers-reduced-motion: reduce)").matches;
B.store={get(k,d){try{const v=localStorage.getItem(k);return v==null?d:JSON.parse(v);}catch(e){return d;}},set(k,v){try{localStorage.setItem(k,JSON.stringify(v));}catch(e){}}};
B.esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
B.softmax=(z,T)=>{T=T||1;const m=Math.max(...z.map(v=>v/T));const e=z.map(v=>Math.exp(v/T-m));const s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);};
B.svg=(tag,attrs,parent)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const k in attrs||{})e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e;};
B.fmt=(x,d)=>Number(x).toFixed(d==null?2:d);
/* theme */
const root=document.documentElement;const th=B.store.get("llmvlm.theme",null);if(th)root.setAttribute("data-theme",th);
document.addEventListener("click",e=>{const t=e.target.closest("[data-theme-toggle]");if(!t)return;
 const dark=root.getAttribute("data-theme")!=="light";
 const n=dark?"light":"dark";root.setAttribute("data-theme",n);B.store.set("llmvlm.theme",n);});
/* math */
B.mathReady=new Promise(res=>{B._mres=res;});
window.MathJax={tex:{inlineMath:[["\\(","\\)"]],displayMath:[["\\[","\\]"]],tags:"ams",macros:{R:"\\mathbb{R}",E:"\\mathbb{E}",softmax:"\\operatorname{softmax}",argmax:"\\operatorname*{arg\\,max}",argmin:"\\operatorname*{arg\\,min}",KL:"\\operatorname{KL}"}},
 svg:{fontCache:"global"},startup:{pageReady:()=>window.MathJax.startup.defaultPageReady().then(()=>B._mres())}};
const mj=document.createElement("script");mj.src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js";mj.async=true;mj.id="MathJax-script";document.head.appendChild(mj);
B.typeset=el=>B.mathReady.then(()=>window.MathJax.typesetPromise(el?[el]:undefined)).catch(()=>{});
/* inline formulas wider than their line become scrollable (only those) */
B.fitMath=()=>{B.$$('mjx-container[jax="SVG"]:not([display="true"])').forEach(m=>{let p=m.parentElement;while(p&&!/^(block|list-item|table-cell|flex|grid)$/.test(getComputedStyle(p).display))p=p.parentElement;if(!p)return;const pw=p.clientWidth;if(!pw)return;m.classList.remove("mjx-wide");if(m.getBoundingClientRect().width>pw+1)m.classList.add("mjx-wide");});};
B.mathReady.then(()=>{B.fitMath();addEventListener("resize",()=>{clearTimeout(B._fm);B._fm=setTimeout(B.fitMath,200);});addEventListener("hashchange",()=>setTimeout(B.fitMath,60));});
/* page features after load */
document.addEventListener("DOMContentLoaded",()=>{
 B.$$(".article").forEach(art=>{
  const scope=art.closest(".pg")||document;const pre=art.dataset.slug?art.dataset.slug+"-":"";
  const hs=B.$$("h2,h3",art).filter(h=>!h.closest(".takeaways,.learn,.fig,.qa,.worked"));let i2=0,i3=0;const items=[];
  hs.forEach(h=>{const pl=h.classList.contains("plain");if(h.tagName==="H2"&&!pl){i2++;i3=0;}else if(h.tagName==="H3"&&!pl)i3++;
   if(!h.id)h.id=pre+(h.tagName==="H2"?"sec-"+i2:"sec-"+i2+"-"+i3);
   items.push({h,sub:h.tagName==="H3",label:(h.tagName==="H2"&&!pl?i2+". ":"")+h.textContent.trim()});});
  const tocL=B.$(".toc-list",scope);
  if(tocL){tocL.innerHTML=items.filter(x=>!x.sub).map(x=>`<li><a href="#${x.h.id}">${B.esc(x.label)}</a></li>`).join("");
   const links=B.$$("a",tocL);const map=new Map(links.map(a=>[a.getAttribute("href").slice(1),a]));
   const io=new IntersectionObserver(es=>{es.forEach(en=>{if(en.isIntersecting){const a=map.get(en.target.id);if(a){links.forEach(x=>x.classList.remove("on"));a.classList.add("on");}}});},{rootMargin:"-15% 0px -75% 0px"});
   items.forEach(x=>io.observe(x.h));
   tocL.addEventListener("click",e=>{if(e.target.closest("a")){const tc=B.$(".toc",scope);if(tc)tc.classList.remove("open");}});}
  const bar=B.$(".readbar",scope);if(bar){const upd=()=>{if(scope.hidden)return;const r=art.getBoundingClientRect();const tot=r.height-innerHeight;bar.style.width=Math.max(0,Math.min(100,100*(-r.top)/Math.max(1,tot)))+"%";};addEventListener("scroll",upd,{passive:true});upd();}
 });
 B.$$(".toc-btn").forEach(tb=>tb.addEventListener("click",()=>{const sc=tb.closest(".pg")||document;const tc=B.$(".toc",sc);if(tc)tc.classList.toggle("open");}));
 B.$$("pre.code").forEach(p=>{const b=document.createElement("button");b.className="copy";b.type="button";b.textContent="Copy";
  b.addEventListener("click",()=>{const t=p.innerText.replace(/^Copy\n?|\nCopy$/g,"");(navigator.clipboard?navigator.clipboard.writeText(p.dataset.raw||t):Promise.reject()).then(()=>{b.textContent="Copied";setTimeout(()=>b.textContent="Copy",1400);}).catch(()=>{b.textContent="Select and copy";});});p.appendChild(b);});
 B.$$("[data-done]").forEach(dk=>{const key="llmvlm.done";const id=dk.dataset.done;
  const upd=()=>{const set=new Set(B.store.get(key,[]));dk.textContent=set.has(id)?"Done. Click to mark as not done":"Mark this session as done";dk.setAttribute("aria-pressed",set.has(id)?"true":"false");};
  dk.addEventListener("click",()=>{const set=new Set(B.store.get(key,[]));set.has(id)?set.delete(id):set.add(id);B.store.set(key,[...set]);upd();});upd();});
});
/* simple step player: el with data-steps=N; calls onStep(i) */
B.stepper=(opts)=>{let i=0,timer=null;const n=opts.n;const prev=opts.prev,next=opts.next,play=opts.play,lab=opts.label;
 const go=k=>{i=Math.max(0,Math.min(n-1,k));opts.onStep(i);if(prev)prev.disabled=i===0;if(next)next.disabled=i===n-1;if(lab)lab.textContent=`Step ${i+1} of ${n}`;};
 const stop=()=>{if(timer){clearInterval(timer);timer=null;if(play)play.textContent="Play";}};
 if(prev)prev.addEventListener("click",()=>{stop();go(i-1);});if(next)next.addEventListener("click",()=>{stop();go(i+1);});
 if(play)play.addEventListener("click",()=>{if(timer){stop();return;}if(i===n-1)go(0);play.textContent="Pause";timer=setInterval(()=>{if(i>=n-1){stop();return;}go(i+1);},opts.ms||2200);});
 go(0);return{go,stop,get i(){return i;}};};
})();
