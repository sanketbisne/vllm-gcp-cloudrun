/**
 * app.js
 * Frontend controller for vLLM on Google Cloud Platform Dashboard
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const statusPulseDot = document.getElementById("statusPulseDot");
  const serviceStatusLabel = document.getElementById("serviceStatusLabel");
  const headerRegion = document.getElementById("headerRegion");
  const gpuBadge = document.getElementById("gpuBadge");
  const btnPingHealth = document.getElementById("btnPingHealth");
  const healthPingMs = document.getElementById("healthPingMs");

  // Tab Elements
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // RAG / Document Elements
  const documentContextText = document.getElementById("documentContextText");
  const ragQueryInput = document.getElementById("ragQueryInput");
  const btnRunStream = document.getElementById("btnRunStream");
  const btnRunExtract = document.getElementById("btnRunExtract");
  const btnReloadSample = document.getElementById("btnReloadSample");
  const terminalOutput = document.getElementById("terminalOutput");
  const modelActiveBadge = document.getElementById("modelActiveBadge");

  // Telemetry Meters
  const metricTTFT = document.getElementById("metricTTFT");
  const metricTPOT = document.getElementById("metricTPOT");
  const metricTPS = document.getElementById("metricTPS");
  const metricTokens = document.getElementById("metricTokens");

  // Guided Extraction
  const extractionContainer = document.getElementById("extractionContainer");
  const extractedCardsGrid = document.getElementById("extractedCardsGrid");
  const btnCopyJson = document.getElementById("btnCopyJson");
  let lastExtractedJson = null;

  // Control Plane elements
  const ctrlServiceUrl = document.getElementById("ctrlServiceUrl");
  const ctrlProjectRegion = document.getElementById("ctrlProjectRegion");
  const ctrlBucketName = document.getElementById("ctrlBucketName");

  // Models catalog container
  const modelsCatalogContainer = document.getElementById("modelsCatalogContainer");
  const stagingCommandSnippet = document.getElementById("stagingCommandSnippet");

  // FinOps elements
  const sliderVolume = document.getElementById("sliderVolume");
  const sliderPromptTokens = document.getElementById("sliderPromptTokens");
  const sliderOutputTokens = document.getElementById("sliderOutputTokens");
  const labelReqVolume = document.getElementById("labelReqVolume");
  const labelPromptTokens = document.getElementById("labelPromptTokens");
  const labelOutputTokens = document.getElementById("labelOutputTokens");
  const costCommercial = document.getElementById("costCommercial");
  const costDedicated = document.getElementById("costDedicated");
  const costCloudRun = document.getElementById("costCloudRun");
  const savingsPercent = document.getElementById("savingsPercent");
  const savingsDollars = document.getElementById("savingsDollars");

  // Snippet elements
  const snippetCurl = document.getElementById("snippetCurl");
  const snippetPython = document.getElementById("snippetPython");

  let appConfig = {};

  // 1. Initialize Tabs
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const targetTab = document.getElementById(btn.dataset.tab);
      if (targetTab) targetTab.classList.add("active");
    });
  });

  // 2. Fetch System Status
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      appConfig = data;

      serviceStatusLabel.textContent = data.status_label || "READY";
      headerRegion.textContent = `GCP ${data.region || "us-central1"}`;
      gpuBadge.textContent = data.accelerator || "1x NVIDIA L4 (24GB)";
      modelActiveBadge.textContent = data.model_name || "qwen-7b";

      if (ctrlServiceUrl) ctrlServiceUrl.textContent = data.service_url;
      if (ctrlProjectRegion) ctrlProjectRegion.textContent = `${data.gcp_project_id} (${data.region})`;
      if (ctrlBucketName) ctrlBucketName.textContent = `gs://${data.bucket_name}/models`;

      if (data.is_live) {
        statusPulseDot.style.backgroundColor = "var(--accent-emerald)";
        statusPulseDot.style.boxShadow = "0 0 10px var(--accent-emerald)";
        healthPingMs.textContent = `${data.health_latency_ms || 35}ms`;
      } else {
        statusPulseDot.style.backgroundColor = "var(--accent-cyan)";
        statusPulseDot.style.boxShadow = "0 0 10px var(--accent-cyan)";
        healthPingMs.textContent = "Demo Mode";
      }

      updateSnippets(data.service_url, data.model_name);
    } catch (e) {
      console.warn("Could not load /api/status:", e);
    }
  }

  // 3. Health Check Ping
  async function pingHealth() {
    healthPingMs.textContent = "...";
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      healthPingMs.textContent = `${data.ping_ms || 42}ms`;
    } catch (e) {
      healthPingMs.textContent = "err";
    }
  }
  btnPingHealth.addEventListener("click", pingHealth);

  // 4. Load Sample Contract Document
  async function loadSampleDoc() {
    try {
      const res = await fetch("/api/sample-doc");
      const data = await res.json();
      documentContextText.value = data.content || "";
    } catch (e) {
      console.error(e);
    }
  }
  btnReloadSample.addEventListener("click", loadSampleDoc);

  // Preset Buttons
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      ragQueryInput.value = btn.dataset.q;
      runStreamChat();
    });
  });

  // 5. Streaming RAG Chat Execution
  async function runStreamChat() {
    const docText = documentContextText.value.trim();
    const query = ragQueryInput.value.trim();

    if (!query) {
      alert("Please enter a question.");
      return;
    }

    btnRunStream.disabled = true;
    btnRunStream.textContent = "Streaming...";
    terminalOutput.textContent = "";
    terminalOutput.classList.add("streaming");
    extractionContainer.style.display = "none";

    // Reset meters
    metricTTFT.textContent = "...";
    metricTPOT.textContent = "...";
    metricTPS.textContent = "...";
    metricTokens.textContent = "0";

    const startTime = performance.now();
    let firstTokenTime = null;
    let tokenCount = 0;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonSafeStringify({ document: docText, query: query }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep remainder

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith("data: ")) continue;
          const dataStr = trimmed.substring(6);

          if (dataStr === "[DONE]") {
            break;
          }

          try {
            const parsed = JSON.parse(dataStr);
            const delta = parsed.choices?.[0]?.delta?.content || "";
            if (delta) {
              if (firstTokenTime === null) {
                firstTokenTime = performance.now();
                const ttft = (firstTokenTime - startTime).toFixed(1);
                metricTTFT.textContent = `${ttft} ms`;
              }
              tokenCount++;
              terminalOutput.textContent += delta;
              terminalOutput.scrollTop = terminalOutput.scrollHeight;
              metricTokens.textContent = tokenCount;

              // Live TPS estimation
              const elapsedSec = (performance.now() - startTime) / 1000;
              if (elapsedSec > 0) {
                metricTPS.textContent = `${(tokenCount / elapsedSec).toFixed(1)} tok/s`;
              }
            }
          } catch (err) {
            // Ignore partial SSE chunk parsing
          }
        }
      }

      // Final metrics calculation
      const totalTime = performance.now() - startTime;
      if (firstTokenTime && tokenCount > 1) {
        const genTime = totalTime - (firstTokenTime - startTime);
        const tpot = (genTime / tokenCount).toFixed(1);
        metricTPOT.textContent = `${tpot} ms`;
        metricTPS.textContent = `${(tokenCount / (totalTime / 1000)).toFixed(1)} tok/s`;
      }
    } catch (e) {
      terminalOutput.textContent += `\n[Stream Error: ${e.message}]`;
    } finally {
      terminalOutput.classList.remove("streaming");
      btnRunStream.disabled = false;
      btnRunStream.textContent = "Ask vLLM";
    }
  }
  btnRunStream.addEventListener("click", runStreamChat);

  // 6. Guided JSON Schema Extraction
  async function runGuidedExtraction() {
    const docText = documentContextText.value.trim();
    btnRunExtract.disabled = true;
    btnRunExtract.textContent = "Extracting...";
    terminalOutput.textContent = "Enforcing Pydantic Schema via vLLM Finite State Machine (FSM) guided decoding...\n";

    try {
      const res = await fetch("/api/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonSafeStringify({ document: docText }),
      });
      const resData = await res.json();

      if (resData.success && resData.data) {
        lastExtractedJson = resData.data;
        terminalOutput.textContent = JSON.stringify(resData.data, null, 2);
        renderExtractionCards(resData.data);
        extractionContainer.style.display = "block";
      } else {
        terminalOutput.textContent = `Extraction failed: ${resData.error || "Unknown error"}`;
      }
    } catch (e) {
      terminalOutput.textContent = `Extraction network error: ${e.message}`;
    } finally {
      btnRunExtract.disabled = false;
      btnRunExtract.textContent = "Run Audit Extraction";
    }
  }
  btnRunExtract.addEventListener("click", runGuidedExtraction);

  function renderExtractionCards(data) {
    extractedCardsGrid.innerHTML = `
      <div class="report-card full-width">
        <div class="title">Agreement Title</div>
        <div class="detail" style="color: var(--accent-cyan);">${escapeHtml(data.document_title || "N/A")}</div>
      </div>
      <div class="report-card">
        <div class="title">Liquidated Breach Indemnity Cap</div>
        <div class="detail" style="color: var(--accent-rose); font-size: 1.15rem;">
          $${(data.max_data_breach_indemnity_usd || 0).toLocaleString()} USD
        </div>
      </div>
      <div class="report-card">
        <div class="title">Guaranteed Monthly Uptime</div>
        <div class="detail" style="color: var(--accent-emerald); font-size: 1.15rem;">
          ${data.sla_availability_percent || 99.95}% Availability
        </div>
      </div>
      <div class="report-card">
        <div class="title">Private VPC Inference Covenant</div>
        <div class="detail" style="color: ${data.private_inference_guarantee ? "var(--accent-emerald)" : "var(--accent-amber)"};">
          ${data.private_inference_guarantee ? "🔒 Zero Third-Party Egress" : "⚠️ Unrestricted"}
        </div>
      </div>
      <div class="report-card">
        <div class="title">Compliance Standards</div>
        <div class="detail" style="font-size: 0.82rem;">
          ${(data.compliance_standards || []).join(", ") || "None specified"}
        </div>
      </div>
      <div class="report-card full-width">
        <div class="title">Risk Assessment Summary</div>
        <div class="detail" style="font-size: 0.85rem; font-weight: normal; color: #cbd5e1;">
          ${escapeHtml(data.risk_summary || "")}
        </div>
      </div>
    `;
  }

  btnCopyJson.addEventListener("click", () => {
    if (lastExtractedJson) {
      navigator.clipboard.writeText(JSON.stringify(lastExtractedJson, null, 2));
      btnCopyJson.textContent = "Copied!";
      setTimeout(() => (btnCopyJson.textContent = "Copy JSON"), 2000);
    }
  });

  // 7. Load Models Catalog
  async function loadModelsCatalog() {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();
      const models = data.models || [];

      modelsCatalogContainer.innerHTML = models
        .map(
          (m) => `
        <div class="model-card ${m.name === (appConfig.model_name || "qwen-7b") ? "active-model" : ""}">
          <div>
            <div class="model-card-header">
              <div class="model-card-title">${escapeHtml(m.name)}</div>
              ${m.recommended ? '<span class="badge badge-cyan">Recommended</span>' : ""}
            </div>
            <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin: 0.25rem 0 0.75rem;">
              ${escapeHtml(m.id)}
            </div>
            <div class="model-spec-row">
              <span>💾 ${escapeHtml(m.size)}</span>
              <span>•</span>
              <span>⚡ ${escapeHtml(m.vram)}</span>
            </div>
            <p style="font-size: 0.82rem; color: var(--text-secondary); margin-top: 0.75rem;">
              ${escapeHtml(m.description)}
            </p>
          </div>
          <button class="btn-secondary select-model-btn" data-model="${escapeHtml(m.id)}" data-name="${escapeHtml(m.name)}" style="width: 100%; justify-content: center; margin-top: 0.5rem;">
            Select & Generate Staging Command
          </button>
        </div>
      `
        )
        .join("");

      document.querySelectorAll(".select-model-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const modelId = btn.dataset.model;
          const modelName = btn.dataset.name;
          const bucket = appConfig.bucket_name || "your-bucket-vllm-models";
          stagingCommandSnippet.textContent = `python sync_model.py --model ${modelId} --bucket ${bucket} --subdir models/${modelName}`;
          alert(`Selected ${modelName}. Staging command updated below!`);
        });
      });
    } catch (e) {
      console.error(e);
    }
  }

  // 8. FinOps Dynamic Calculation
  function updateFinOps() {
    const volume = parseInt(sliderVolume.value, 10);
    const promptTokens = parseInt(sliderPromptTokens.value, 10);
    const outputTokens = parseInt(sliderOutputTokens.value, 10);

    labelReqVolume.textContent = `${volume.toLocaleString()} requests`;
    labelPromptTokens.textContent = `${promptTokens.toLocaleString()} tokens`;
    labelOutputTokens.textContent = `${outputTokens.toLocaleString()} tokens`;

    // 1. Commercial Closed SaaS pricing ($2.50 / 1M prompt, $10.00 / 1M output)
    const promptCost = (volume * promptTokens * 2.5) / 1_000_000;
    const outputCost = (volume * outputTokens * 10.0) / 1_000_000;
    const totalCommercial = Math.round(promptCost + outputCost);

    // 2. Dedicated GKE 1x L4 node ($0.95/hr * 720 hours = $684/mo fixed)
    const totalDedicated = 684;

    // 3. Cloud Run Serverless GPU
    // Assume average 50 tokens/sec throughput -> active inference time per req = outputTokens / 50 seconds.
    // L4 GPU on Cloud Run is ~$0.00028 per second of active compute.
    // Scale-to-zero means 0 idle costs.
    const activeSecPerReq = Math.max(0.8, outputTokens / 45);
    const totalComputeSeconds = volume * activeSecPerReq;
    const cloudRunComputeCost = totalComputeSeconds * 0.00028;
    const gcsStorageCost = 5; // $5/mo for model storage
    const totalCloudRun = Math.round(cloudRunComputeCost + gcsStorageCost);

    costCommercial.textContent = `$${totalCommercial.toLocaleString()} / mo`;
    costDedicated.textContent = `$${totalDedicated.toLocaleString()} / mo`;
    costCloudRun.textContent = `$${totalCloudRun.toLocaleString()} / mo`;

    // Savings percentage vs Commercial
    const savings = Math.max(0, totalCommercial - totalCloudRun);
    const pct = totalCommercial > 0 ? Math.round((savings / totalCommercial) * 100) : 0;

    savingsPercent.textContent = `${pct}% SAVINGS`;
    savingsDollars.textContent = `Save ~$${savings.toLocaleString()}/mo vs third-party commercial APIs`;
  }

  sliderVolume.addEventListener("input", updateFinOps);
  sliderPromptTokens.addEventListener("input", updateFinOps);
  sliderOutputTokens.addEventListener("input", updateFinOps);

  // 9. Update Code Snippets
  function updateSnippets(url, model) {
    const targetUrl = url || "https://vllm-l4-server-preview.a.run.app";
    const targetModel = model || "qwen-7b";

    snippetCurl.textContent = `curl -X POST "${targetUrl}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer \${VLLM_API_KEY}" \\
  -d '{
    "model": "${targetModel}",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain PagedAttention in 2 sentences."}
    ],
    "stream": true
  }'`;

    snippetPython.textContent = `from openai import OpenAI

client = OpenAI(
    base_url="${targetUrl}/v1",
    api_key="\${VLLM_API_KEY}"
)

stream = client.chat.completions.create(
    model="${targetModel}",
    messages=[{"role": "user", "content": "Hello vLLM on GCP Cloud Run!"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`;
  }

  function jsonSafeStringify(obj) {
    return JSON.stringify(obj);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial Load Sequence
  fetchStatus().then(() => {
    loadSampleDoc();
    loadModelsCatalog();
    updateFinOps();
  });
});
