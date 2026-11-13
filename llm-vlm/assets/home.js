(function(){const B=window.Book;
document.addEventListener("pointermove",e=>{const c=e.target.closest&&e.target.closest(".track");B.$$(".track").forEach(t=>{if(t!==c){t.style.removeProperty("--mx");t.style.removeProperty("--my");t.style.removeProperty("--rx");t.style.removeProperty("--ry");}});
 if(!c||B.reduced)return;const r=c.getBoundingClientRect();const x=(e.clientX-r.left)/r.width-.5,y=(e.clientY-r.top)/r.height-.5;
 c.style.setProperty("--mx",x.toFixed(3));c.style.setProperty("--my",y.toFixed(3));c.style.setProperty("--rx",(x*5).toFixed(2)+"deg");c.style.setProperty("--ry",(-y*5).toFixed(2)+"deg");});
document.addEventListener("DOMContentLoaded",()=>{const done=new Set(B.store.get("llmvlm.done",[]));
 B.$$("[data-prog]").forEach(el=>{const ids=el.dataset.prog.split(",").filter(Boolean);const n=ids.filter(i=>done.has(i)).length;const tot=+el.dataset.total;
  const bar=el.querySelector("i");if(bar)bar.style.width=(100*n/Math.max(1,tot))+"%";const lab=el.parentElement.querySelector("[data-plab]");if(lab)lab.textContent=`${n}/${tot} done`;});
 B.$$("[data-ready]").forEach(el=>{if(done.has(el.dataset.ready)){el.textContent="Done";el.className="st2 ready";}});});
})();
