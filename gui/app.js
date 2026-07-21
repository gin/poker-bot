/* poker-bot replay GUI — vanilla JS over /api/*. */

const $ = (sel) => document.querySelector(sel);
const state = { run: null, order: "recent", tag: "", activeHand: null };

const SUITS = { s: "♠", h: "♥", d: "♦", c: "♣" };

function cardsHtml(text) {
  if (!text) return "";
  return text
    .trim()
    .split(/[\s,]+/)
    .filter(Boolean)
    .map((c) => {
      const rank = c.slice(0, -1).toUpperCase();
      const suit = c.slice(-1).toLowerCase();
      const red = suit === "h" || suit === "d";
      return `<span class="card ${red ? "red" : ""}">${rank}${SUITS[suit] || suit}</span>`;
    })
    .join("");
}

function netHtml(net) {
  if (net == null) return `<span class="net muted">?</span>`;
  const cls = net >= 0 ? "pos" : "neg";
  return `<span class="net ${cls}">${net >= 0 ? "+" : ""}${net}</span>`;
}

function tagsHtml(tags) {
  return (tags || [])
    .map((t) => {
      const [kind, name] = t.split(":");
      return `<span class="tag ${kind}">${name}</span>`;
    })
    .join(" ");
}

function highlightTags(message) {
  return (message || "").replace(
    /\[(guard|exploit):([a-z0-9_-]+)\]/g,
    (_, kind, name) => `<span class="tag ${kind}">${name}</span>`
  );
}

async function api(path, params = {}) {
  const url = new URL(path, location.origin);
  Object.entries(params).forEach(([k, v]) => v && url.searchParams.set(k, v));
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

async function loadRuns() {
  const runs = await api("/api/runs");
  const sel = $("#run-select");
  sel.innerHTML = runs
    .map(
      (r) =>
        `<option value="${r.run_id}">${r.strategy} vs ${r.opponent || "?"} · ` +
        `${r.hands} hands · net ${r.net_chips >= 0 ? "+" : ""}${r.net_chips} · ${r.started_at}</option>`
    )
    .join("");
  state.run = runs.length ? runs[0].run_id : null;
  if (state.run) await Promise.all([loadHands(), loadTags()]);
}

async function loadTags() {
  const tags = await api("/api/tags", { run_id: state.run });
  $("#tag-list").innerHTML = tags
    .map((t) => `<option value="${t.tag}"></option>`)
    .join("");
  $("#tag-summary").textContent = tags
    .slice(0, 4)
    .map((t) => `${t.tag}×${t.fires}${t.avg_net != null ? ` (${t.avg_net > 0 ? "+" : ""}${t.avg_net}/hand)` : ""}`)
    .join("  ");
}

async function loadHands() {
  const hands = await api("/api/hands", {
    run_id: state.run,
    order: state.order,
    tag: state.tag,
  });
  const STREETS = ["", "pre", "flop", "turn", "river"];
  $("#hand-list").innerHTML = hands
    .map(
      (h) => `
      <div class="hand-row" data-hand="${h.hand_id}">
        ${netHtml(h.net)}
        <span>${cardsHtml(h.hole_cards)}</span>
        <span class="muted">${h.position || ""} ${h.players || ""}p → ${STREETS[h.street_depth] || ""}</span>
        ${tagsHtml(h.tags)}
      </div>`
    )
    .join("") || `<div class="placeholder">no hands match</div>`;
  document.querySelectorAll(".hand-row").forEach((el) =>
    el.addEventListener("click", () => loadHand(el.dataset.hand, el))
  );
}

async function loadHand(handId, rowEl) {
  document.querySelectorAll(".hand-row.active").forEach((el) => el.classList.remove("active"));
  if (rowEl) rowEl.classList.add("active");
  const hand = await api("/api/hand", { run_id: state.run, hand_id: handId });

  const blocks = hand.timeline
    .map(
      (street) => `
      <div class="street-block">
        <div class="street-head">
          <span>${street.street}</span>
          <span>${cardsHtml(street.board)}</span>
        </div>
        ${street.events
          .map((e) =>
            e.kind === "hero"
              ? `<div class="event hero">
                   <span class="who">HERO ${cardsHtml(e.hole_cards)}</span>
                   <span class="action ${e.action}">${e.action}${e.amount ? " " + e.amount : ""}</span>
                   <span class="muted">pot ${e.pot ?? "?"} · stack ${e.hero_stack ?? "?"}</span>
                   <span class="msg">${highlightTags(e.message)}</span>
                 </div>`
              : `<div class="event">
                   <span class="who">${e.who}</span>
                   <span class="action ${e.action}">${e.action}${e.amount ? " " + e.amount : ""}</span>
                   <span class="muted">pot ${e.pot ?? ""}</span>
                 </div>`
          )
          .join("")}
      </div>`
    )
    .join("");

  const revealed = (hand.opponents || []).filter((o) => o.revealed_hole_cards);
  const revealedHtml = revealed.length
    ? `<div class="revealed">
         <span class="revealed-label">revealed</span>
         ${revealed
           .map(
             (o) =>
               `<span class="reveal-row"><span class="who">${o.handle || o.agent_id.slice(0, 10)}</span>${cardsHtml(o.revealed_hole_cards)}</span>`
           )
           .join("")}
         ${(hand.winners || [])
           .filter((w) => w.handName || w.message)
           .map(
             (w) =>
               `<span class="muted win-row">${w.agentName || "?"} won${w.amount != null ? " " + w.amount : ""}${w.handName ? " · " + w.handName : ""}${w.message ? " — " + w.message : ""}</span>`
           )
           .join("")}
       </div>`
    : "";

  $("#replay").innerHTML = `
    <div class="hand-header">
      ${netHtml(hand.hero.net)}
      <span>${cardsHtml(hand.hero.hole_cards)}</span>
      <span class="muted">${hand.hero.position || ""} · ${hand.hand_id}</span>
    </div>
    ${revealedHtml}
    ${blocks}`;

  $("#opponents").innerHTML =
    hand.opponents
      .map((o) => {
        const api_ = o.api_stats || {};
        const ps = api_.playingStyle || {};
        const style = ps.label;
        const revealed = o.revealed_hole_cards
          ? `<div class="opp-revealed">${cardsHtml(o.revealed_hole_cards)}${o.payout_chips != null ? `<span class="muted"> ${o.payout_chips >= 0 ? "+" : ""}${o.payout_chips}</span>` : ""}</div>`
          : "";
        const archetype = ps.archetype
          ? `<div class="api-note">${ps.archetype}${ps.tagline ? ` — ${ps.tagline}` : ""}</div>`
          : "";
        return `
        <div class="opp-card">
          <h3>${o.handle || o.agent_id.slice(0, 12)}</h3>
          ${revealed}
          ${archetype}
          <dl class="statgrid">
            <dt>observed hands</dt><dd>${o.hands_seen}</dd>
            <dt>VPIP / PFR</dt><dd>${o.vpip_pct ?? "?"}% / ${o.pfr_pct ?? "?"}%</dd>
            <dt>aggression</dt><dd>${o.aggression_pct ?? "?"}%</dd>
            <dt>call freq</dt><dd>${o.call_pct ?? "?"}%</dd>
            <dt>fold to bet</dt><dd>${o.fold_to_bet_pct ?? "?"}%</dd>
            <dt>showdowns won</dt><dd>${o.won_showdown}/${o.showdowns}</dd>
            ${api_.sampleSize ? `
              <dt>API sample</dt><dd>${api_.sampleSize}</dd>
              <dt>API VPIP / bluff</dt>
              <dd>${Math.round((api_.vpip || 0) * 100)}% / ${Math.round((api_.bluffPct || 0) * 100)}%</dd>` : ""}
          </dl>
          ${style ? `<div class="api-note">API style: ${style}</div>` : ""}
        </div>`;
      })
      .join("") || `<div class="placeholder">no opponent data</div>`;
}

$("#run-select").addEventListener("change", (e) => {
  state.run = e.target.value;
  loadHands();
  loadTags();
});
$("#order-select").addEventListener("change", (e) => {
  state.order = e.target.value;
  loadHands();
});
$("#tag-filter").addEventListener("input", (e) => {
  state.tag = e.target.value.trim();
  loadHands();
});

loadRuns().catch((err) => {
  $("#replay").innerHTML = `<div class="placeholder">failed to load: ${err}</div>`;
});
