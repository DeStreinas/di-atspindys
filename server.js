import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const HOST = "0.0.0.0";
const PORT = 4173;

const GROQ_CHAT_MODEL = "openai/gpt-oss-20b";
const GROQ_STT_MODEL = "whisper-large-v3-turbo";

const VOICE_DIR = path.join(os.homedir(), "di-atspindys-voice");
const REGINUTE_DIR = path.join(VOICE_DIR, "reginute");
const PYTHON = path.join(VOICE_DIR, ".venv", "bin", "python");
const VOICE_MODEL = path.join(
  REGINUTE_DIR, "models", "lt_LT-reginute1-medium.onnx"
);
const VOICE_CONFIG = VOICE_MODEL + ".json";
const VOICE_SCRIPT = path.join(REGINUTE_DIR, "demo_piper_wheel.py");

const systemInstruction = `
Tu esi DI Atspindys — asmeninis, mandagus ir natūraliai
lietuviškai kalbantis AI asistentas išmaniajame veidrodyje.

- Atsakyk lietuviškai, nebent naudotojas prašo kitaip.
- Kalbėk natūraliai, aiškiai ir glaustai, nes atsakymai bus
  perskaitomi balsu.
- Venk Markdown, lentelių ir ilgų sąrašų.
- Jei nežinai, sąžiningai tai pasakyk.
- Neapsimesk, kad valdai įrenginius ar matai duomenis,
  kurių iš tikrųjų negavai.
`;

function send(res, status, data, type = "application/json; charset=utf-8") {
  res.writeHead(status, {
    "Content-Type": type,
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff"
  });
  res.end(data);
}

function json(res, status, data) {
  send(res, status, JSON.stringify(data));
}

async function readBody(req, limit) {
  const chunks = [];
  let size = 0;

  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) {
      const error = new Error("Užklausa per didelė.");
      error.status = 413;
      throw error;
    }
    chunks.push(chunk);
  }

  return Buffer.concat(chunks);
}

async function readJson(req, limit = 100_000) {
  const raw = await readBody(req, limit);
  try {
    return JSON.parse(raw.toString("utf8"));
  } catch {
    const error = new Error("Netinkamas JSON formatas.");
    error.status = 400;
    throw error;
  }
}

async function groqChat(req, res) {
  if (!process.env.GROQ_API_KEY) {
    return json(res, 503, {
      error: "Nesukonfigūruotas Groq API raktas."
    });
  }

  const body = await readJson(req);
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return json(res, 400, { error: "Trūksta žinutės." });
  }

  const messages = [{
    role: "system",
    content: systemInstruction
  }];

  for (const message of body.messages.slice(-20)) {
    if (!["user", "assistant"].includes(message.role)) continue;

    const content = String(message.text ?? "").trim().slice(0, 4000);
    if (content) messages.push({ role: message.role, content });
  }

  if (messages.length === 1) {
    return json(res, 400, { error: "Žinutė tuščia." });
  }

  const response = await fetch(
    "https://api.groq.com/openai/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.GROQ_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: GROQ_CHAT_MODEL,
        messages,
        temperature: 0.7,
        max_completion_tokens: 400
      }),
      signal: AbortSignal.timeout(45_000)
    }
  );

  const data = await response.json();

  if (!response.ok) {
    console.error("Groq chat error:", response.status, data.error?.message);
    return json(res, 502, {
      error: "AI užklausa nepavyko. Patikrink Groq API prieigą."
    });
  }

  const answer = data.choices?.[0]?.message?.content?.trim();
  if (!answer) {
    return json(res, 502, { error: "AI grąžino tuščią atsakymą." });
  }

  return json(res, 200, { answer });
}

async function transcribe(req, res) {
  if (!process.env.GROQ_API_KEY) {
    return json(res, 503, {
      error: "Nesukonfigūruotas Groq API raktas."
    });
  }

  const contentType = req.headers["content-type"] || "";
  if (!/^audio\/(webm|ogg|wav|mpeg|mp4|mp4a-latm|x-wav|flac)/i.test(contentType)) {
    return json(res, 415, {
      error: "Nepalaikomas garso formatas."
    });
  }

  const audio = await readBody(req, 20 * 1024 * 1024);
  if (!audio.length) {
    return json(res, 400, { error: "Gautas tuščias garso įrašas." });
  }

  const extension = contentType.includes("ogg") ? "ogg"
    : contentType.includes("wav") ? "wav"
    : contentType.includes("mp4") ? "mp4"
    : contentType.includes("mpeg") ? "mp3"
    : contentType.includes("flac") ? "flac"
    : "webm";

  const form = new FormData();
  form.append("file", new Blob([audio], { type: contentType }),
    `recording.${extension}`);
  form.append("model", GROQ_STT_MODEL);
  form.append("language", "lt");
  form.append("response_format", "json");
  form.append("temperature", "0");

  const response = await fetch(
    "https://api.groq.com/openai/v1/audio/transcriptions",
    {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.GROQ_API_KEY}`
      },
      body: form,
      signal: AbortSignal.timeout(60_000)
    }
  );

  const data = await response.json();

  if (!response.ok) {
    console.error("Groq transcription error:", response.status,
      data.error?.message);
    return json(res, 502, {
      error: "Nepavyko atpažinti kalbos. Bandyk dar kartą."
    });
  }

  const text = String(data.text || "").trim();
  if (!text) {
    return json(res, 422, { error: "Kalbos įraše nepavyko atpažinti žodžių." });
  }

  return json(res, 200, { text });
}

async function synthesize(req, res) {
  const body = await readJson(req, 20_000);
  const text = String(body.text || "").trim();

  if (!text) {
    return json(res, 400, { error: "Trūksta teksto sintezei." });
  }
  if (text.length > 3000) {
    return json(res, 413, { error: "Tekstas per ilgas sintezei." });
  }

  for (const file of [PYTHON, VOICE_SCRIPT, VOICE_MODEL, VOICE_CONFIG]) {
    if (!fs.existsSync(file)) {
      console.error("Reginutė file missing:", file);
      return json(res, 503, {
        error: "Reginutės balso failai nerasti."
      });
    }
  }

  const tempDir = await fs.promises.mkdtemp(
    path.join(os.tmpdir(), "atspindys-")
  );
  const wavPath = path.join(tempDir, `${randomUUID()}.wav`);

  try {
    await new Promise((resolve, reject) => {
      const child = spawn(PYTHON, [
        VOICE_SCRIPT,
        VOICE_MODEL,
        VOICE_CONFIG,
        text,
        wavPath
      ], {
        cwd: REGINUTE_DIR,
        stdio: ["ignore", "ignore", "pipe"]
      });

      let stderr = "";
      child.stderr.setEncoding("utf8");
      child.stderr.on("data", chunk => {
        stderr = (stderr + chunk).slice(-4000);
      });

      child.on("error", reject);
      child.on("close", code => {
        if (code === 0) resolve();
        else reject(new Error(`Reginutė exit ${code}: ${stderr}`));
      });
    });

    const wav = await fs.promises.readFile(wavPath);
    return send(res, 200, wav, "audio/wav");
  } catch (error) {
    console.error("Reginutė synthesis failed:", error.message);
    return json(res, 500, {
      error: "Nepavyko sugeneruoti Reginutės balso."
    });
  } finally {
    await fs.promises.rm(tempDir, { recursive: true, force: true });
  }
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${HOST}:${PORT}`);
    const pathname = url.pathname;

    if (req.method === "POST" && pathname === "/api/chat") {
      return await groqChat(req, res);
    }

    if (req.method === "POST" && pathname === "/api/transcribe") {
      return await transcribe(req, res);
    }

    if (req.method === "POST" && pathname === "/api/tts") {
      return await synthesize(req, res);
    }

    if (req.method !== "GET" && req.method !== "HEAD") {
      return send(res, 405, "Method not allowed", "text/plain; charset=utf-8");
    }

    if (pathname !== "/" && pathname !== "/index.html") {
      return send(res, 404, "Not found", "text/plain; charset=utf-8");
    }

    fs.readFile(path.join(ROOT, "index.html"), (error, content) => {
      if (error) {
        return send(res, 500, "Could not load interface",
          "text/plain; charset=utf-8");
      }
      send(res, 200, req.method === "HEAD" ? "" : content,
        "text/html; charset=utf-8");
    });
  } catch (error) {
    if (error.status === 413) {
      return json(res, 413, { error: error.message });
    }
    if (error.status === 400) {
      return json(res, 400, { error: error.message });
    }
    console.error("Request failed:", error.message);
    if (!res.headersSent) {
      json(res, 500, { error: "Vidinė serverio klaida." });
    } else {
      res.end();
    }
  }
});

server.listen(PORT, HOST, () => {
  console.log(`DI Atspindys: http://${HOST}:${PORT}`);
  console.log(`Groq API key: ${process.env.GROQ_API_KEY ? "configured" : "MISSING"}`);
  console.log("Routes: /api/transcribe, /api/chat, /api/tts");
});
