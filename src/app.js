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
        // Ignore opens within 90s of send: Gmail/mail servers often fetch images on delivery (scanning),
        // which would falsely mark as opened before the user actually opened the email.
        const sentTime = doc.createdAt || new Date();
        const secondsSinceSent = (Date.now() - new Date(sentTime).getTime()) / 1000;
        if (secondsSinceSent >= 90) {
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

// Email click tracking: redirect through backend to record click, then redirect to destination
const CLICK_GRACE_SECONDS = 90;
const FALLBACK_REDIRECT = "https://www.google.com";

app.get("/track/click/:id", async (req, res) => {
  const id = req.params.id;
  const rawUrl = req.query.url;
  let destination = typeof rawUrl === "string" ? decodeURIComponent(rawUrl.trim()) : "";
  if (!destination || !/^https?:\/\//i.test(destination)) {
    destination = FALLBACK_REDIRECT;
  }
  // Append email id so landing page can report "credentials entered" for this email
  const sep = destination.indexOf("?") >= 0 ? "&" : "?";
  destination = `${destination}${sep}e=${encodeURIComponent(id)}`;
  try {
    const doc = await Email.findById(id);
    if (doc) {
      const wasAlreadyClicked = !!doc.clickedAt;
      if (!wasAlreadyClicked) {
        const sentTime = doc.createdAt || new Date();
        const secondsSinceSent = (Date.now() - new Date(sentTime).getTime()) / 1000;
        if (secondsSinceSent >= CLICK_GRACE_SECONDS) {
          doc.clickedAt = new Date();
          await doc.save();
          if (doc.campaignId) {
            await Campaign.updateOne(
              { _id: doc.campaignId },
              {
                $inc: { "stats.totalEmailClicked": 1 },
                $set: {
                  "targetUsers.$[elem].emailStatus": "clicked",
                  "targetUsers.$[elem].emailClickedAt": doc.clickedAt,
                },
              },
              { arrayFilters: [{ "elem.email": doc.sentTo }] }
            );
          }
          console.log("Email click recorded", { id, sentTo: doc.sentTo });
        }
      }
    }
  } catch (err) {
    console.error("Email click tracking DB update failed:", err.message);
  }
  res.redirect(302, destination);
});

// Email credentials-entered tracking (landing page form submit – no credentials stored, just that user submitted)
// CORS preflight: OPTIONS must be handled so the landing page (different origin) can POST
app.options("/track/credentials", (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.set("Access-Control-Allow-Headers", "Content-Type");
  res.status(204).end();
});

app.post("/track/credentials", express.json(), async (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");
  const emailId = (req.body && req.body.emailId) || (req.query && req.query.e);
  if (!emailId) {
    return res.status(400).json({ success: false, message: "Missing emailId" });
  }
  try {
    const doc = await Email.findById(emailId);
    if (!doc) {
      return res.status(404).json({ success: false, message: "Email not found" });
    }
    const alreadyRecorded = !!doc.credentialsEnteredAt;
    if (!alreadyRecorded) {
      doc.credentialsEnteredAt = new Date();
      await doc.save();
      if (doc.campaignId) {
        await Campaign.updateOne(
          { _id: doc.campaignId },
          {
            $inc: { "stats.totalEmailReported": 1 },
            $set: {
              "targetUsers.$[elem].emailStatus": "reported",
              "targetUsers.$[elem].emailReportedAt": doc.credentialsEnteredAt,
            },
          },
          { arrayFilters: [{ "elem.email": doc.sentTo }] }
        );
      }
      console.log("Email credentials entered recorded", { id: emailId, sentTo: doc.sentTo });
    }
    return res.status(200).json({ success: true, recorded: !alreadyRecorded });
  } catch (err) {
    console.error("Credentials tracking failed:", err.message);
    return res.status(500).json({ success: false, message: "Server error" });
  }
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
