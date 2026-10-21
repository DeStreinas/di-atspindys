
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

const HOST = "127.0.0.1";
const PORT = 4173;
const MODEL = "gemini-2.5-flash";
const ROOT = __dirname;

const systemInstruction = `
Tu esi DI Atspindys — asmeninis, mandagus ir natūraliai
lietuviškai kalbantis AI asistentas išmaniajame veidrodyje.

- Visada atsakyk lietuviškai, nebent naudotojas prašo kitaip.
- Kalbėk natūraliai, aiškiai ir glaustai, nes atsakymai bus
  perskaitomi balsu.
- Venk Markdown formatavimo, lentelių ir ilgų sąrašų.
- Jei nežinai, sąžiningai tai pasakyk.
- Neapsimesk, kad valdai įrenginius ar matai duomenis,
  kurių iš tikrųjų negavai.
`;

const server = http.createServer(async (req, res) => {
  res.setHeader("Cache-Control", "no-store");

  if (req.method === "POST" && req.url === "/api/chat") {
    const key = process.env.GEMINI_API_KEY;

    if (!key) {
      res.writeHead(503, { "Content-Type": "application/json" });
      return res.end(JSON.stringify({
        error: "Nesukonfigūruotas Gemini API raktas."
      }));
    }

    try {
      let raw = "";

      for await (const chunk of req) {
        raw += chunk;

        if (raw.length > 100_000) {
          res.writeHead(413);
          return res.end();
        }
      }

      const body = JSON.parse(raw);

      if (!Array.isArray(body.messages) || body.messages.length === 0) {
        res.writeHead(400, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({
          error: "Trūksta žinutės."
        }));
      }

      const contents = body.messages.slice(-20).map(message => ({
        role: message.role === "model" ? "model" : "user",
        parts: [{
          text: String(message.text || "").slice(0, 4000)
        }]
      }));

      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-goog-api-key": key
          },
          body: JSON.stringify({
            systemInstruction: {
              parts: [{ text: systemInstruction }]
            },
            contents,
            generationConfig: {
              temperature: 0.7,
              maxOutputTokens: 400
            }
          }),
          signal: AbortSignal.timeout(45000)
        }
      );

      const data = await response.json();

      if (!response.ok) {
        console.error("Gemini API error:", response.status, data);

        res.writeHead(502, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({
          error: "Gemini užklausa nepavyko. Patikrink API raktą ir modelį."
        }));
      }

      const answer = data.candidates?.[0]?.content?.parts
        ?.map(part => part.text || "")
        .join("")
        .trim();

      if (!answer) {
        throw new Error("Gemini returned an empty response");
      }

      res.writeHead(200, { "Content-Type": "application/json" });
      return res.end(JSON.stringify({ answer }));
    } catch (error) {
      console.error("Chat error:", error.message);

      res.writeHead(500, { "Content-Type": "application/json" });
      return res.end(JSON.stringify({
        error: "Nepavyko gauti atsakymo. Bandyk dar kartą."
      }));
    }
  }

  if (req.method !== "GET" && req.method !== "HEAD") {
    res.writeHead(405);
    return res.end("Method not allowed");
  }

  const requested = new URL(req.url, `http://${HOST}:${PORT}`).pathname;

  if (requested !== "/" && requested !== "/index.html") {
    res.writeHead(404);
    return res.end("Not found");
  }

  fs.readFile(path.join(ROOT, "index.html"), (error, content) => {
    if (error) {
      res.writeHead(500);
      return res.end("Could not load interface");
    }

    res.writeHead(200, {
      "Content-Type": "text/html; charset=utf-8"
    });

    res.end(content);
  });
});

server.listen(PORT, HOST, () => {
  console.log(`DI Atspindys: http://${HOST}:${PORT}`);
});

