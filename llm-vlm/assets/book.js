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
const root=document.documentElement;const th=B.store.get("llmvlm.theme2",null);if(th)root.setAttribute("data-theme",th);
document.addEventListener("click",e=>{const t=e.target.closest("[data-theme-toggle]");if(!t)return;
 const dark=root.getAttribute("data-theme")!=="light";
 const n=dark?"light":"dark";root.setAttribute("data-theme",n);B.store.set("llmvlm.theme2",n);});
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
 B.$$("pre.code").forEach(p=>B.highlight(p));
 B.$$("pre.code").forEach(p=>{const b=document.createElement("button");b.className="copy";b.type="button";b.textContent="Copy";
  b.addEventListener("click",()=>{const t=p.innerText.replace(/^Copy\n?|\nCopy$/g,"");(navigator.clipboard?navigator.clipboard.writeText(p.dataset.raw||t):Promise.reject()).then(()=>{b.textContent="Copied";setTimeout(()=>b.textContent="Copy",1400);}).catch(()=>{b.textContent="Select and copy";});});p.appendChild(b);});
 B.$$("[data-done]").forEach(dk=>{const key="llmvlm.done";const id=dk.dataset.done;
  const upd=()=>{const set=new Set(B.store.get(key,[]));dk.textContent=set.has(id)?"Done. Click to mark as not done":"Mark this session as done";dk.setAttribute("aria-pressed",set.has(id)?"true":"false");};
  dk.addEventListener("click",()=>{const set=new Set(B.store.get(key,[]));set.has(id)?set.delete(id):set.add(id);B.store.set(key,[...set]);upd();B.courseProgress();});upd();});
 /* per-reader course completion (stored only in this browser) */
 B.$$(".topbar-in").forEach(tb=>{if(B.$(".cprog",tb))return;const a=document.createElement("a");a.className="cprog";a.href="/#curriculum";a.title="Your course progress (saved in this browser)";
  a.innerHTML='<span class="cprog-lab">Your progress</span><span class="cprog-bar"><i></i></span><b class="cprog-n"></b>';const sp=B.$(".tb-sp",tb);if(sp)sp.after(a);else tb.appendChild(a);});
 B.courseProgress();
 addEventListener("storage",e=>{if(e.key==="llmvlm.done")B.courseProgress();});
});
B.TOTAL=239;
B.courseProgress=()=>{const n=new Set(B.store.get("llmvlm.done",[])).size,pc=Math.min(100,100*n/B.TOTAL);
 B.$$(".cprog").forEach(c=>{const i=B.$(".cprog-bar i",c);if(i)i.style.width=pc.toFixed(1)+"%";const t=B.$(".cprog-n",c);if(t)t.textContent=n+" / "+B.TOTAL;c.setAttribute("aria-label","Course progress: "+n+" of "+B.TOTAL+" lessons done");});};
/* simple step player: el with data-steps=N; calls onStep(i) */
B.stepper=(opts)=>{let i=0,timer=null;const n=opts.n;const prev=opts.prev,next=opts.next,play=opts.play,lab=opts.label;
 const go=k=>{i=Math.max(0,Math.min(n-1,k));opts.onStep(i);if(prev)prev.disabled=i===0;if(next)next.disabled=i===n-1;if(lab)lab.textContent=`Step ${i+1} of ${n}`;};
 const stop=()=>{if(timer){clearInterval(timer);timer=null;if(play)play.textContent="Play";}};
 if(prev)prev.addEventListener("click",()=>{stop();go(i-1);});if(next)next.addEventListener("click",()=>{stop();go(i+1);});
 if(play)play.addEventListener("click",()=>{if(timer){stop();return;}if(i===n-1)go(0);play.textContent="Pause";timer=setInterval(()=>{if(i>=n-1){stop();return;}go(i+1);},opts.ms||2200);});
 go(0);return{go,stop,get i(){return i;}};};
/* syntax highlighting for pre.code: detects python / bash / json / output, labels the block, colors tokens */
(function(){
 const esc=s=>s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
 const sp=(c,t)=>`<span class="tk-${c}">${esc(t)}</span>`;
 const KW=new Set("and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case".split(" "));
 const CONST=new Set(["True","False","None","self","cls","__name__"]);
 const BI=new Set("print len range enumerate zip int float str list dict set tuple min max sum abs sorted map filter isinstance super open round any all type iter next reversed bool bytes hasattr getattr setattr format repr slice divmod pow hash id input vars dir chr ord".split(" "));
 function detect(t){
  if(/^\s*(import |from \S+ import |def |class |@\w|for .+:\s*$|if .+:\s*$|while .+:\s*$|with .+:\s*$|print\(|return\b|[A-Za-z_][\w.]*\s*=\s*\S)/m.test(t)&&!/^\s*(\$ |pip |sudo |cd |export |curl |git |nvidia-smi|torchrun |vllm serve|docker |tritonserver|python3? -m )/m.test(t.split("\n").find(l=>l.trim()&&!l.trim().startsWith("#"))||""))return "python";
  if(/^\s*(\$ |pip3? |sudo |cd |export |curl |wget |git |nvidia-smi|torchrun |vllm |docker |tritonserver|python3? |huggingface-cli|ollama |llama-|make |ls |mkdir |source |conda )/m.test(t))return "bash";
  const s=t.trim();if(/^[\[{]/.test(s)&&/["\]}]\s*$/.test(s)&&/"[^"]*"\s*:/.test(s))return "json";
  return "output";}
 function py(t){
  const re=/(#[^\n]*)|([rRbBfFuU]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(^[ \t]*@[\w.]+)|(\b(?:0[xX][\da-fA-F_]+|\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?j?)\b|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_]\w*)|(\s+)|([\s\S])/gm;
  let out="",m,prev="",prevSig="";
  while((m=re.exec(t))){const [tok,cm,str,dec,num,word,ws,ch]=m;
   if(cm)out+=sp("cm",tok);else if(str)out+=sp("str",tok);else if(dec)out+=sp("dec",tok);else if(num)out+=sp("num",tok);
   else if(word){const rest=t.slice(re.lastIndex),call=/^\s*\(/.test(rest);
    if(prevSig==="def")out+=sp("def",tok);else if(prevSig==="class")out+=sp("cls",tok);
    else if(KW.has(tok))out+=sp("kw",tok);else if(CONST.has(tok))out+=sp("const",tok);
    else if(BI.has(tok)&&prev!==".")out+=sp("bi",tok);
    else if(call)out+=sp("fn",tok);else if(/^[A-Z][A-Za-z0-9]*[a-z]/.test(tok))out+=sp("cls",tok);else out+=esc(tok);}
   else if(ws)out+=tok;else out+=/[=+\-*/%<>!&|^~@]/.test(tok)?sp("op",tok):esc(tok);
   if(!ws){prev=tok;prevSig=(word&&(tok==="def"||tok==="class"))?tok:"";}}
  return out;}
 function sh(t){
  return t.split("\n").map(line=>{let out="",first=true;const re=/(#[^\n]*)|("(?:\\.|[^"\\])*"|'[^']*')|(\$\{?\w+\}?)|(\s--?[\w-]+(?:=\S*)?)|(\s+)|([^\s"'#$]+)|([\s\S])/g;let m;
   while((m=re.exec(line))){const [tok,cm,str,v,flag,ws,word]=m;
    if(cm&&(first||/\s$/.test(line.slice(0,m.index))))out+=sp("cm",tok);else if(str)out+=sp("str",tok);else if(v)out+=sp("const",tok);else if(flag)out+=sp("num",tok);
    else if(ws)out+=tok;else if(word){if(first&&tok!=="$"){out+=sp("fn",tok);first=false;}else if(tok==="$")out+=sp("op",tok);else if(/^(&&|\|\||\||;)$/.test(tok)){out+=sp("op",tok);first=true;}else out+=esc(tok);}
    else out+=esc(tok);}
   return out;}).join("\n");}
 function js(t){
  return esc(t).replace(/(\/\/[^\n]*)|(&quot;|")((?:\\.|[^"\\\n])*)"(\s*:)?|\b(true|false|null)\b|(-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)/g,(all,cm,q,s,colon,lit,num)=>cm?`<span class="tk-cm">${cm}</span>`:lit?`<span class="tk-const">${lit}</span>`:num?`<span class="tk-num">${num}</span>`:colon?`<span class="tk-fn">"${s}"</span>${colon}`:`<span class="tk-str">"${s}"</span>`);}
 Book.highlight=p=>{
  if(p.dataset.hl||p.querySelector("button,input,select,textarea"))return;
  const raw=p.textContent;const lang=(p.dataset.lang||detect(raw)).toLowerCase();
  p.dataset.lang=lang;p.dataset.hl="1";if(!p.dataset.raw)p.dataset.raw=raw;
  if(lang==="python"||lang==="py")p.innerHTML=py(raw);else if(lang==="bash"||lang==="sh"||lang==="shell")p.innerHTML=sh(raw);else if(lang==="json")p.innerHTML=js(raw);
  p.classList.add("lang-"+lang);};
})();
})();
