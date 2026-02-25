const { sendMail: sendMailWithTracking, expressApp } = require("nodemailer-mail-tracking");
const Email = require("../models/Email");
const Campaign = require("../models/Campaign");

// Per-recipient data passed to getData at send time so the JWT contains emailRecordId/campaignId
// Key normalized to lowercase so library's recipient (from envelope) matches our setPendingTracking(recipient)
const pendingTracking = new Map();

function normalizeRecipient(recipient) {
  if (recipient == null) return "";
  return String(recipient).toLowerCase().trim();
}

function getMailTrackingOptions() {
  const baseUrl = (process.env.BACKEND_URL || `http://localhost:${process.env.PORT || 5001}`).replace(/\/$/, "") + "/api/track/email";
  console.log("[Mail tracking] getMailTrackingOptions() called", { baseUrl, BACKEND_URL: process.env.BACKEND_URL || "(not set)" });
  const jwtSecret = process.env.MAIL_TRACK_JWT_SECRET || process.env.JWT_SECRET || "cybershield-mail-track-secret";

  return {
    baseUrl,
    jwtSecret,
    // API: getData(data) – default data is { recipient }. Return { ...data, ...yourData } so JWT contains recipient + our ids
    getData(data) {
      const key = normalizeRecipient(data.recipient);
      const extra = pendingTracking.get(key) || {};
      const out = { ...data, ...extra };
      console.log("[Mail tracking] getData called", { recipient: data.recipient, key, hasExtra: Object.keys(extra).length > 0, keys: Object.keys(extra) });
      if (Object.keys(extra).length === 0) {
        console.warn("[Mail tracking] getData: no pending data for recipient – ensure setPendingTracking(recipient, ...) was called with same recipient before sendMail", { recipient: data.recipient, pendingKeys: Array.from(pendingTracking.keys()) });
      }
      return out;
    },
    async onBlankImageView(data) {
      const { emailRecordId, campaignId, recipient } = data;
      console.log("[Mail tracking] step: onBlankImageView received", { emailRecordId, campaignId, recipient });
      if (!emailRecordId) {
        console.log("[Mail tracking] step: onBlankImageView skipped (no emailRecordId)");
        return;
      }
      try {
        const emailRecord = await Email.findById(emailRecordId);
        if (!emailRecord) {
          console.log("[Mail tracking] step: onBlankImageView email record not found", { emailRecordId });
          return;
        }
        if (emailRecord.openedAt) {
          console.log("[Mail tracking] step: onBlankImageView already opened (idempotent)", { emailRecordId });
          return;
        }
        const now = new Date();
        emailRecord.openedAt = now;
        await emailRecord.save();
        console.log("[Mail tracking] step: onBlankImageView saved openedAt to Email", { emailRecordId });
        if (campaignId) {
          const campaign = await Campaign.findById(campaignId);
          if (campaign) {
            const target = campaign.targetUsers.find((t) => (t.email || "").toLowerCase() === (recipient || "").toLowerCase());
            if (target && target.emailStatus !== "opened" && target.emailStatus !== "clicked") {
              target.emailStatus = "opened";
              target.emailOpenedAt = now;
              campaign.stats.totalEmailOpened = (campaign.stats.totalEmailOpened || 0) + 1;
              await campaign.save();
              console.log("[Mail tracking] step: onBlankImageView campaign target updated", { campaignId, totalEmailOpened: campaign.stats.totalEmailOpened });
            } else {
              console.log("[Mail tracking] step: onBlankImageView campaign target not updated", { targetFound: !!target, targetStatus: target?.emailStatus });
            }
          } else {
            console.log("[Mail tracking] step: onBlankImageView campaign not found", { campaignId });
          }
        }
      } catch (err) {
        console.error("[Mail tracking] onBlankImageView error", err);
      }
    },
    async onLinkClick(data) {
      const { emailRecordId, campaignId, recipient } = data;
      const linkPreview = data.link ? data.link.substring(0, 70) + (data.link.length > 70 ? "..." : "") : "";
      console.log("[Mail tracking] step: onLinkClick received", { emailRecordId, campaignId, recipient, link: linkPreview });
      if (!emailRecordId) {
        console.log("[Mail tracking] step: onLinkClick skipped (no emailRecordId)");
        return;
      }
      try {
        const emailRecord = await Email.findById(emailRecordId);
        if (!emailRecord) {
          console.log("[Mail tracking] step: onLinkClick email record not found", { emailRecordId });
          return;
        }
        if (emailRecord.clickedAt) {
          console.log("[Mail tracking] step: onLinkClick already clicked (idempotent)", { emailRecordId });
          return;
        }
        const now = new Date();
        emailRecord.clickedAt = now;
        await emailRecord.save();
        console.log("[Mail tracking] step: onLinkClick saved clickedAt to Email", { emailRecordId });
        if (campaignId) {
          const campaign = await Campaign.findById(campaignId);
          if (campaign) {
            const target = campaign.targetUsers.find((t) => (t.email || "").toLowerCase() === (recipient || "").toLowerCase());
            if (target) {
              target.emailStatus = "clicked";
              target.emailClickedAt = now;
              campaign.stats.totalEmailClicked = (campaign.stats.totalEmailClicked || 0) + 1;
              await campaign.save();
              console.log("[Mail tracking] step: onLinkClick campaign target updated", { campaignId, totalEmailClicked: campaign.stats.totalEmailClicked });
            } else {
              console.log("[Mail tracking] step: onLinkClick target not found for recipient", { recipient });
            }
          } else {
            console.log("[Mail tracking] step: onLinkClick campaign not found", { campaignId });
          }
        }
      } catch (err) {
        console.error("[Mail tracking] onLinkClick error", err);
      }
    },
  };
}

function setPendingTracking(recipient, data) {
  const key = normalizeRecipient(recipient);
  pendingTracking.set(key, data);
  console.log("[Mail tracking] step: setPendingTracking", { recipient, key, emailRecordId: data.emailRecordId, campaignId: data.campaignId });
}

function clearPendingTracking(recipient) {
  const key = normalizeRecipient(recipient);
  pendingTracking.delete(key);
  console.log("[Mail tracking] step: clearPendingTracking", { recipient, key });
}

module.exports = {
  sendMailWithTracking,
  expressApp,
  getMailTrackingOptions,
  setPendingTracking,
  clearPendingTracking,
};
