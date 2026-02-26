const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
require("dotenv").config();

// Import routes
const adminRoutes = require("./routes/admin");
const orgRoutes = require("./routes/orgs");
const userRoutes = require("./routes/users");
const whatsappCampaignRoutes = require("./routes/whatsappCampaigns");
const voicePhishingRoutes = require("./routes/voicePhishing");
const emailRoutes = require("./routes/email");
const emailTemplateRoutes = require("./routes/emailTemplates");
const whatsAppTemplateRoutes = require("./routes/whatsAppTemplates");
const voicePhishingTemplateRoutes = require("./routes/voicePhishingTemplates");
const campaignRoutes = require("./routes/campaigns");
const incidentRoutes = require("./routes/incidents");
const chatRoutes = require("./routes/chat");
const courseRoutes = require("./routes/courses");
const certificateRoutes = require("./routes/certificates");
const uploadRoutes = require("./routes/upload");
const Email = require("./models/Email");
const Campaign = require("./models/Campaign");

const app = express();

// 1x1 transparent GIF for email open tracking (Buffer method – no file read)
const TRACKING_PIXEL_GIF = Buffer.from([
  0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00,
  0x80, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x2c,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02,
  0x02, 0x44, 0x01, 0x00, 0x3b,
]);

// Security middleware
app.use(helmet());

// CORS configuration
app.use(
  cors({
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    credentials: true,
  })
);

// Body parsing middleware
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// Root – so visiting the backend URL shows something instead of 404
app.get("/", (req, res) => {
  res.json({
    message: "CyberShield Backend API",
    health: "/health",
    docs: "Use /api/* routes (e.g. /api/campaigns, /api/users)",
  });
});

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({
    status: "OK",
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || "development",
  });
});

// Email open tracking: middleware logs the request, then handler serves 1x1 GIF
app.use(
  "/track/open/:id",
  (req, res, next) => {
    if (req.method !== "GET") return next();
    console.log("Email open tracking: request for image", { id: req.params.id });
    next();
  },
  async (req, res, next) => {
    if (req.method !== "GET") return next();
    const id = req.params.id;
    try {
      const doc = await Email.findById(id);
      if (!doc) {
        next();
        return;
      }
      const wasAlreadyOpened = !!doc.openedAt;
      if (!wasAlreadyOpened) {
        doc.openedAt = new Date();
        await doc.save();
        if (doc.campaignId) {
          await Campaign.updateOne(
            { _id: doc.campaignId },
            {
              $inc: { "stats.totalEmailOpened": 1 },
              $set: {
                "targetUsers.$[elem].emailStatus": "opened",
                "targetUsers.$[elem].emailOpenedAt": doc.openedAt,
              },
            },
            { arrayFilters: [{ "elem.email": doc.sentTo }] }
          );
        }
        console.log("Email open recorded", { id, sentTo: doc.sentTo });
      }
    } catch (err) {
      console.error("Email open tracking DB update failed:", err.message);
    }
    next();
  }
);
app.get("/track/open/:id", (req, res) => {
  res.writeHead(200, { "Content-Type": "image/gif" });
  res.end(TRACKING_PIXEL_GIF, "binary");
});

// API routes
app.use("/api/admins", adminRoutes);
app.use("/api/orgs", orgRoutes);
app.use("/api/users", userRoutes);
app.use("/api/whatsapp-campaigns", whatsappCampaignRoutes);
app.use("/api/voice-phishing", voicePhishingRoutes);
app.use("/api/email-campaigns", emailRoutes);
app.use("/api/email-templates", emailTemplateRoutes);
app.use("/api/whatsapp-templates", whatsAppTemplateRoutes);
app.use("/api/voice-phishing-templates", voicePhishingTemplateRoutes);
app.use("/api/campaigns", campaignRoutes);
app.use("/api/incidents", incidentRoutes);
app.use("/api/chat", chatRoutes);
app.use("/api/courses", courseRoutes);
app.use("/api/certificates", certificateRoutes);
app.use("/api/upload", uploadRoutes);

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: "Route not found" });
});

// Global error handler
app.use((error, req, res, next) => {
  console.error("Global error handler:", error);

  // Multer errors
  if (error.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({ error: "File too large. Maximum size is 100MB." });
  }

  if (error.message === "Only CSV files are allowed") {
    return res.status(400).json({ error: "Only CSV files are allowed" });
  }

  // Default error response
  res.status(500).json({
    error: "Internal server error",
    message:
      process.env.NODE_ENV === "development"
        ? error.message
        : "Something went wrong",
  });
});

module.exports = app;
