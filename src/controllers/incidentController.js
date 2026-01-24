const mlPhishingService = require("../services/mlPhishingService");

/**
 * Analyze a reported incident (email or WhatsApp) using ML pipeline only.
 * POST /api/incidents/analyze
 * Body: { messageType, message, subject?, from?, urls?, date?, text? }
 */
async function analyzeIncident(req, res) {
  try {
    const body = req.body || {};
    const messageType = (body.messageType || "email").toLowerCase();
    const message = body.message || body.text || "";

    if (!message || typeof message !== "string") {
      return res.status(400).json({
        success: false,
        error: "Missing required field: message",
        is_phishing: null,
      });
    }

    const urls = Array.isArray(body.urls) ? body.urls : body.urls ? [body.urls] : [];
    const reportData = {
      messageType: messageType === "whatsapp" ? "whatsapp" : "email",
      message,
      text: body.text || message,
      subject: body.subject || "",
      from: body.from || "",
      from_phone: body.from_phone || body.from || "",
      urls,
      date: body.date || new Date().toISOString(),
      timestamp: body.timestamp || body.date || new Date().toISOString(),
    };

    const formatted = mlPhishingService.formatIncidentForML(reportData);
    const result = await mlPhishingService.predictIncident(formatted);

    return res.status(200).json({
      success: result.success,
      is_phishing: result.is_phishing,
      phishing_probability: result.phishing_probability,
      legitimate_probability: result.legitimate_probability,
      confidence: result.confidence,
      error: result.error || undefined,
      persuasion_cues: result.persuasion_cues,
    });
  } catch (err) {
    console.error("Incident analyze error:", err);
    return res.status(500).json({
      success: false,
      error: err.message || "Analysis failed",
      is_phishing: null,
    });
  }
}

module.exports = { analyzeIncident };
