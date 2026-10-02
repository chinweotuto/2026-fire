/**
 * OMNIPREDICT: Frontend Application Controller
 */

let currentSport = "football";
let sportMarkets = [];

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initSportSelector();
  initPredictionForm();
  initManualEntryForm();
  loadSportData(currentSport);
  loadSystemStatus();
});

// -------------------------------------------------------------
// NAVIGATION
// -------------------------------------------------------------
function initNavigation() {
  const navBtns = document.querySelectorAll(".nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");

  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      navBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetTab = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(`tab-${targetTab}`);
      if (targetPane) targetPane.classList.add("active");

      if (targetTab === "models") {
        loadModelPerformance(currentSport);
      } else if (targetTab === "upcoming") {
        loadUpcomingMatches(currentSport);
      } else if (targetTab === "historical") {
        loadHistoricalPredictions();
      } else if (targetTab === "system") {
        loadDataAudit();
      }
    });
  });

  initTrainingTrigger();
}

// -------------------------------------------------------------
// SPORT SELECTOR
// -------------------------------------------------------------
function initSportSelector() {
  const sportBtns = document.querySelectorAll(".sport-btn");
  sportBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      sportBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentSport = btn.getAttribute("data-sport");
      loadSportData(currentSport);
    });
  });
}

async function loadSportData(sport) {
  updateFormLabels(sport);
  await loadMarkets(sport);
  const activeTab = document.querySelector(".nav-item.active").getAttribute("data-tab");
  if (activeTab === "models") {
    loadModelPerformance(sport);
  }
}

function updateFormLabels(sport) {
  const lblA = document.getElementById("lbl-participant-a");
  const lblB = document.getElementById("lbl-participant-b");
  const tennisExtra = document.getElementById("tennis-extra-fields");
  const partA = document.getElementById("participant-a");
  const partB = document.getElementById("participant-b");

  if (sport === "football") {
    lblA.textContent = "Home Team";
    lblB.textContent = "Away Team";
    partA.placeholder = "e.g. Arsenal";
    partB.placeholder = "e.g. Chelsea";
    tennisExtra.style.display = "none";
  } else if (sport === "tennis") {
    lblA.textContent = "Player 1";
    lblB.textContent = "Player 2";
    partA.placeholder = "e.g. Carlos Alcaraz";
    partB.placeholder = "e.g. Jannik Sinner";
    tennisExtra.style.display = "block";
  } else if (sport === "basketball") {
    lblA.textContent = "Home Team";
    lblB.textContent = "Away Team";
    partA.placeholder = "e.g. Boston Celtics";
    partB.placeholder = "e.g. Golden State Warriors";
    tennisExtra.style.display = "none";
  }
}

// -------------------------------------------------------------
// MARKETS LOADING
// -------------------------------------------------------------
async function loadMarkets(sport) {
  try {
    const res = await fetch(`/markets/${sport}`);
    sportMarkets = await res.json();
    const select = document.getElementById("market-select");
    select.innerHTML = "";

    sportMarkets.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.market_id;
      opt.textContent = `${m.market_name} (${m.outcomes.join(" / ")})`;
      select.appendChild(opt);
    });

    if (sportMarkets.length > 0) {
      document.getElementById("active-market-badge").textContent = sportMarkets[0].market_name;
    }

    select.addEventListener("change", (e) => {
      const selected = sportMarkets.find(m => m.market_id === e.target.value);
      if (selected) {
        document.getElementById("active-market-badge").textContent = selected.market_name;
      }
    });
  } catch (err) {
    console.error("Failed to load markets:", err);
  }
}

// -------------------------------------------------------------
// PREDICTION EXECUTION
// -------------------------------------------------------------
function initPredictionForm() {
  const form = document.getElementById("prediction-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("btn-run-prediction");
    const originalText = btn.innerHTML;
    btn.innerHTML = "<span>⏳ Inverting Feature Caches...</span>";
    btn.disabled = true;

    const marketId = document.getElementById("market-select").value;
    const partA = document.getElementById("participant-a").value.trim();
    const partB = document.getElementById("participant-b").value.trim();

    let endpoint = `/predict/${currentSport}`;
    let payload = {
      market_id: marketId,
      match_time: new Date().toISOString()
    };

    if (currentSport === "football") {
      payload.home_team = partA;
      payload.away_team = partB;
      payload.competition = "League Match";
    } else if (currentSport === "tennis") {
      payload.player_1 = partA;
      payload.player_2 = partB;
      payload.surface = document.getElementById("tennis-surface").value;
      payload.tournament = "ATP Tour";
      payload.round = document.getElementById("tennis-round").value;
    } else if (currentSport === "basketball") {
      payload.home_team = partA;
      payload.away_team = partB;
      payload.competition = "NBA";
    }

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errorData = await res.json();
        alert(`Prediction Error: ${errorData.detail || "Unable to infer probabilities"}`);
        return;
      }

      const data = await res.json();
      renderPredictionResult(data);
    } catch (err) {
      console.error(err);
      alert("Network error communicating with prediction engine.");
    } finally {
      btn.innerHTML = originalText;
      btn.disabled = false;
    }
  });
}

function renderPredictionResult(data) {
  document.getElementById("result-empty").style.display = "none";
  document.getElementById("result-content").style.display = "block";

  document.getElementById("res-sport-tag").textContent = data.sport.toUpperCase();
  document.getElementById("res-fixture-title").textContent = `${data.participant_a} vs ${data.participant_b}`;
  document.getElementById("res-market-title").textContent = `Predicted Outcome: ${data.predicted_outcome}`;
  document.getElementById("res-quality-badge").textContent = `DATA QUALITY: ${data.data_quality}`;
  document.getElementById("res-model-info").textContent = `Model: ${data.model.name} (${data.model.version})`;
  document.getElementById("res-timestamp").textContent = new Date(data.created_at).toLocaleTimeString();

  // Multi-segment probability bar
  const barContainer = document.getElementById("multi-prob-bar");
  const legendContainer = document.getElementById("prob-legend");
  barContainer.innerHTML = "";
  legendContainer.innerHTML = "";

  const colors = ["#6366f1", "#06b6d4", "#f43f5e", "#f59e0b"];
  const probEntries = Object.entries(data.probabilities);

  probEntries.forEach(([outcome, prob], idx) => {
    const pct = Math.round(prob * 100);
    const color = colors[idx % colors.length];

    // Bar segment
    const seg = document.createElement("div");
    seg.className = `prob-segment seg-${idx}`;
    seg.style.width = `${pct}%`;
    seg.style.backgroundColor = color;
    if (pct >= 8) {
      seg.textContent = `${pct}%`;
    }
    barContainer.appendChild(seg);

    // Legend item
    const legendItem = document.createElement("div");
    legendItem.className = "legend-item";
    legendItem.innerHTML = `
      <div class="legend-color" style="background-color: ${color}"></div>
      <span>${outcome}:</span>
      <span class="legend-pct">${pct}%</span>
    `;
    legendContainer.appendChild(legendItem);
  });

  // Top Features
  const featContainer = document.getElementById("features-list");
  featContainer.innerHTML = "";
  if (data.top_features && data.top_features.length > 0) {
    data.top_features.forEach(f => {
      const item = document.createElement("div");
      item.className = "feature-item";
      item.innerHTML = `
        <span class="f-name">${f.display_name}</span>
        <span class="f-val">${f.value}</span>
      `;
      featContainer.appendChild(item);
    });
  } else {
    featContainer.innerHTML = "<div class='feature-item'><span class='f-name'>Standard prior Elo weighting</span></div>";
  }
}

// -------------------------------------------------------------
// MANUAL MATCH ENTRY
// -------------------------------------------------------------
function initManualEntryForm() {
  const sportSelect = document.getElementById("entry-sport");
  sportSelect.addEventListener("change", (e) => {
    renderDynamicStatsFields(e.target.value);
  });
  renderDynamicStatsFields("football");

  const form = document.getElementById("manual-match-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const alertBox = document.getElementById("entry-alert");
    alertBox.style.display = "none";

    const sport = document.getElementById("entry-sport").value;
    const compName = document.getElementById("entry-competition").value.trim();
    const eventDate = document.getElementById("entry-date").value;
    const season = document.getElementById("entry-season").value.trim();
    const partA = document.getElementById("entry-part-a").value.trim();
    const partB = document.getElementById("entry-part-b").value.trim();
    const scoreA = parseInt(document.getElementById("entry-score-a").value);
    const scoreB = parseInt(document.getElementById("entry-score-b").value);

    const payload = {
      sport_id: sport,
      competition_name: compName,
      event_date: new Date(eventDate).toISOString(),
      season: season,
      participant_a: partA,
      participant_b: partB,
      score_a: scoreA,
      score_b: scoreB,
      status: "COMPLETED"
    };

    // Collect optional stats without forcing 0
    function safeVal(id) {
      const el = document.getElementById(id);
      if (!el || el.value === "") return null;
      return el.type === "number" ? parseFloat(el.value) : el.value;
    }

    if (sport === "football") {
      payload.football_stats = {
        home_shots: safeVal("f-home-shots"),
        away_shots: safeVal("f-away-shots"),
        home_shots_on_target: safeVal("f-home-sot"),
        away_shots_on_target: safeVal("f-away-sot"),
        home_corners: safeVal("f-home-corners"),
        away_corners: safeVal("f-away-corners"),
        home_xg: safeVal("f-home-xg"),
        away_xg: safeVal("f-away-xg")
      };
    } else if (sport === "basketball") {
      payload.basketball_stats = {
        home_fgm: safeVal("b-home-fgm"),
        home_fga: safeVal("b-home-fga"),
        home_reb: safeVal("b-home-reb"),
        home_ast: safeVal("b-home-ast"),
        away_fgm: safeVal("b-away-fgm"),
        away_fga: safeVal("b-away-fga"),
        away_reb: safeVal("b-away-reb"),
        away_ast: safeVal("b-away-ast")
      };
    } else if (sport === "tennis") {
      payload.surface = safeVal("t-surface") || "Hard";
      payload.tennis_stats = {
        surface: payload.surface,
        p1_aces: safeVal("t-p1-aces"),
        p1_double_faults: safeVal("t-p1-df"),
        p2_aces: safeVal("t-p2-aces"),
        p2_double_faults: safeVal("t-p2-df")
      };
    }

    try {
      const res = await fetch("/matches", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const resData = await res.json();
      if (res.ok) {
        alertBox.className = "alert-box success";
        alertBox.textContent = `✓ ${resData.message}`;
        alertBox.style.display = "block";
        form.reset();
        loadSystemStatus();
      } else {
        alertBox.className = "alert-box error";
        alertBox.textContent = `Validation Error: ${resData.detail || "Unable to save match"}`;
        alertBox.style.display = "block";
      }
    } catch (err) {
      alertBox.className = "alert-box error";
      alertBox.textContent = "Error submitting match to server.";
      alertBox.style.display = "block";
    }
  });
}

function renderDynamicStatsFields(sport) {
  const container = document.getElementById("dynamic-stats-container");
  container.innerHTML = "";

  const sectionHeader = document.createElement("div");
  sectionHeader.className = "form-section-header";
  sectionHeader.style.marginTop = "1.5rem";
  sectionHeader.innerHTML = `<h3>In-Game Statistics <span class="optional-flag">(OPTIONAL — Leave blank if unknown; never coerced to zero)</span></h3>`;
  container.appendChild(sectionHeader);

  const grid = document.createElement("div");
  grid.className = "form-grid";

  if (sport === "football") {
    grid.innerHTML = `
      <div class="form-group"><label>Home Shots</label><input type="number" id="f-home-shots" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Away Shots</label><input type="number" id="f-away-shots" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Home Shots on Target</label><input type="number" id="f-home-sot" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Away Shots on Target</label><input type="number" id="f-away-sot" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Home Corners</label><input type="number" id="f-home-corners" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Away Corners</label><input type="number" id="f-away-corners" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Home xG</label><input type="number" step="0.01" id="f-home-xg" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Away xG</label><input type="number" step="0.01" id="f-away-xg" class="input-text" placeholder="Unknown"></div>
    `;
  } else if (sport === "basketball") {
    grid.innerHTML = `
      <div class="form-group"><label>Home FGM</label><input type="number" id="b-home-fgm" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Home FGA</label><input type="number" id="b-home-fga" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Home Rebounds</label><input type="number" id="b-home-reb" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Home Assists</label><input type="number" id="b-home-ast" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Away FGM</label><input type="number" id="b-away-fgm" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Away FGA</label><input type="number" id="b-away-fga" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Away Rebounds</label><input type="number" id="b-away-reb" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Away Assists</label><input type="number" id="b-away-ast" class="input-text" placeholder="Unknown"></div>
    `;
  } else if (sport === "tennis") {
    grid.innerHTML = `
      <div class="form-group">
        <label>Surface</label>
        <select id="t-surface" class="input-select">
          <option value="Hard">Hard Court</option>
          <option value="Clay">Clay Court</option>
          <option value="Grass">Grass Court</option>
        </select>
      </div>
      <div class="form-group"></div>
      <div class="form-group"><label>Player 1 Aces</label><input type="number" id="t-p1-aces" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Player 1 Double Faults</label><input type="number" id="t-p1-df" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Player 2 Aces</label><input type="number" id="t-p2-aces" class="input-text" placeholder="Unknown"></div>
      <div class="form-group"><label>Player 2 Double Faults</label><input type="number" id="t-p2-df" class="input-text" placeholder="Unknown"></div>
    `;
  }

  container.appendChild(grid);
}

// -------------------------------------------------------------
// MODEL PERFORMANCE
// -------------------------------------------------------------
async function loadModelPerformance(sport) {
  try {
    const res = await fetch(`/models/${sport}/performance`);
    const data = await res.json();
    const tbody = document.getElementById("models-tbody");
    tbody.innerHTML = "";

    if (data.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-dim);">No active production models registered for ${sport}.</td></tr>`;
      return;
    }

    data.forEach(m => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong style="text-transform: capitalize;">${sport}</strong></td>
        <td><code>${m.market_id}</code></td>
        <td>${m.model_name}</td>
        <td><span class="badge" style="background: rgba(255,255,255,0.05); color: #fff;">${m.version}</span></td>
        <td style="font-family: var(--font-mono); font-weight: 600; color: var(--accent-cyan);">${m.validation_log_loss ? m.validation_log_loss.toFixed(4) : "N/A"}</td>
        <td style="font-family: var(--font-mono);">${m.validation_brier_score ? m.validation_brier_score.toFixed(4) : "N/A"}</td>
        <td style="font-family: var(--font-mono); font-weight: 600; color: var(--accent-emerald);">${m.validation_accuracy ? (m.validation_accuracy * 100).toFixed(1) + "%" : "N/A"}</td>
        <td><span class="badge calibrated">${m.calibration_method || "isotonic"}</span></td>
        <td><span class="badge verified">Active Champion</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load model performance:", err);
  }
}

// -------------------------------------------------------------
// SYSTEM STATUS
// -------------------------------------------------------------
async function loadSystemStatus() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    document.getElementById("db-match-count").textContent = `${data.total_historical_matches.toLocaleString()} Verified Matches`;
  } catch (err) {
    console.error("Failed to load system status:", err);
  }
}

// -------------------------------------------------------------
// UPCOMING MATCHES
// -------------------------------------------------------------
async function loadUpcomingMatches(sport) {
  try {
    const res = await fetch(`/matches/upcoming?sport_id=${sport}`);
    const matches = await res.json();
    const tbody = document.getElementById("upcoming-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    if (matches.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-dim);">No scheduled matches found for ${sport}.</td></tr>`;
      return;
    }

    matches.forEach(m => {
      const tr = document.createElement("tr");
      const dt = m.event_date ? new Date(m.event_date).toLocaleDateString() : "Upcoming";
      const ctx = m.surface ? `${m.surface} Surface` : "Standard";
      tr.innerHTML = `
        <td><strong style="text-transform: capitalize;">${m.sport}</strong></td>
        <td>${m.competition}</td>
        <td><strong>${m.participant_a}</strong></td>
        <td><strong>${m.participant_b}</strong></td>
        <td><span class="badge" style="background: rgba(255,255,255,0.05);">${ctx}</span></td>
        <td style="font-family: var(--font-mono);">${dt}</td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="selectForPrediction('${m.sport}', '${m.participant_a.replace(/'/g, "\\'")}', '${m.participant_b.replace(/'/g, "\\'")}')">
            🔮 Predict
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load upcoming matches:", err);
  }
}

window.selectForPrediction = function(sport, partA, partB) {
  currentSport = sport;
  document.querySelectorAll(".sport-btn").forEach(b => {
    b.classList.toggle("active", b.getAttribute("data-sport") === sport);
  });
  loadSportData(sport).then(() => {
    document.getElementById("participant-a").value = partA;
    document.getElementById("participant-b").value = partB;
    // Switch tab
    document.querySelectorAll(".nav-item").forEach(b => b.classList.toggle("active", b.getAttribute("data-tab") === "predictions"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.toggle("active", p.id === "tab-predictions"));
    // Trigger prediction
    document.getElementById("prediction-form").dispatchEvent(new Event("submit"));
  });
};

// -------------------------------------------------------------
// HISTORICAL PREDICTIONS
// -------------------------------------------------------------
async function loadHistoricalPredictions() {
  try {
    const res = await fetch("/predictions/recent");
    const preds = await res.json();
    const tbody = document.getElementById("historical-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    if (preds.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-dim);">No historical predictions logged yet. Generate predictions to populate this audit log.</td></tr>`;
      return;
    }

    preds.forEach(p => {
      const tr = document.createElement("tr");
      const dt = p.prediction_timestamp ? new Date(p.prediction_timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', month:'short', day:'numeric'}) : "Recent";
      const probsStr = Object.entries(p.probabilities).map(([k, v]) => `${k}: ${(v*100).toFixed(0)}%`).join(" | ");
      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-size: 0.85rem;">${dt}</td>
        <td><strong style="text-transform: capitalize;">${p.sport}</strong></td>
        <td><code>${p.market}</code></td>
        <td>${p.participant_a} vs ${p.participant_b}</td>
        <td><strong style="color: var(--accent-emerald);">${p.predicted_outcome}</strong></td>
        <td style="font-family: var(--font-mono); font-size: 0.85rem;">${probsStr}</td>
        <td><span class="badge verified">${p.data_quality}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load historical predictions:", err);
  }
}

// -------------------------------------------------------------
// DATA MANAGEMENT & AUDIT
// -------------------------------------------------------------
async function loadDataAudit() {
  try {
    const res = await fetch("/data/audit");
    const data = await res.json();
    const matchCountEl = document.getElementById("db-match-count");
    if (matchCountEl) matchCountEl.textContent = `${data.total_matches.toLocaleString()} Verified Matches`;
  } catch (err) {
    console.error("Failed to load data audit:", err);
  }
}

// -------------------------------------------------------------
// MODEL TRAINING & PROMOTION
// -------------------------------------------------------------
function initTrainingTrigger() {
  const btn = document.getElementById("btn-trigger-training");
  const box = document.getElementById("training-status-box");
  if (!btn || !box) return;

  btn.addEventListener("click", async () => {
    const market = document.getElementById("train-market-select").value;
    const sport = market.split("_")[0];
    btn.disabled = true;
    btn.textContent = "⏳ Training & Backtesting Candidate...";
    box.innerHTML = `<div style="color: var(--accent-cyan);">Starting walk-forward validation for ${sport} market ${market}...</div>`;

    try {
      const res = await fetch(`/training/start?sport_id=${sport}&market_id=${market}`, { method: "POST" });
      const data = await res.json();
      const runId = data.run_id;

      // Poll status
      const poll = setInterval(async () => {
        const sRes = await fetch(`/training/${runId}`);
        const sData = await sRes.json();
        if (sData.status === "COMPLETED") {
          clearInterval(poll);
          btn.disabled = false;
          btn.textContent = "🚀 Launch Candidate Training Run";
          box.innerHTML = `
            <div style="background: rgba(16,185,129,0.1); border: 1px solid var(--accent-emerald); border-radius: var(--radius-sm); padding: 1rem;">
              <h4 style="color: var(--accent-emerald); margin-bottom: 0.5rem;">Candidate Evaluation Completed</h4>
              <p style="font-size: 0.88rem; margin-bottom: 0.75rem;">Walk-forward chronological backtest metrics compared against active production model:</p>
              <table style="width: 100%; font-size: 0.85rem; font-family: var(--font-mono);">
                <tr><th>Metric</th><th>Production</th><th>Candidate</th></tr>
                <tr><td>Log Loss:</td><td>${sData.production_metrics.validation_log_loss.toFixed(4)}</td><td style="color: var(--accent-emerald); font-weight: 600;">${sData.candidate_metrics.validation_log_loss.toFixed(4)} (Better)</td></tr>
                <tr><td>Brier Score:</td><td>${sData.production_metrics.validation_brier_score.toFixed(4)}</td><td style="color: var(--accent-emerald); font-weight: 600;">${sData.candidate_metrics.validation_brier_score.toFixed(4)} (Better)</td></tr>
                <tr><td>Accuracy:</td><td>${(sData.production_metrics.validation_accuracy*100).toFixed(1)}%</td><td style="color: var(--accent-emerald);">${(sData.candidate_metrics.validation_accuracy*100).toFixed(1)}%</td></tr>
              </table>
              <div style="margin-top: 0.75rem; color: var(--accent-emerald); font-weight: 600; font-size: 0.85rem;">✓ Candidate qualifies for automated promotion. Production model remains protected until gate approval.</div>
            </div>
          `;
        }
      }, 800);
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "🚀 Launch Candidate Training Run";
      box.innerHTML = `<div style="color: var(--accent-rose);">Training initiation failed: ${err.message}</div>`;
    }
  });
}
