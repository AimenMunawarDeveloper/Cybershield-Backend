const mongoose = require("mongoose");
const WhatsAppRiskEvent = require("../models/WhatsAppRiskEvent");
const User = require("../models/User");
const { updateUserCombinedLearningScore } = require("./combinedLearningScoreService");

// User model field name — do NOT use "whatsappRiskScore"; only this field is valid.
const USER_FIELD_LEARNING_SCORE_WHATSAPP = "learningScoreWhatsapp";

const WHATSAPP_RISK_DECAY_RATE = 0.95; // Each day's influence decreases by 5%
const ELIGIBLE_ROLES = ["affiliated", "non_affiliated"];
// Max possible raw score (read 0.2 + click 0.5 + credentials 0.7) — same as email, used to normalize
const MAX_RAW_SCORE = 0.2 + 0.5 + 0.7; // 1.4

function isEligibleForWhatsAppRiskScoring(role) {
  return role && ELIGIBLE_ROLES.includes(role);
}

function getWhatsAppRiskWeight(eventType) {
  const weights = {
    whatsapp_read: 0.2,
    whatsapp_clicked: 0.5,
    whatsapp_credentials_submitted: 0.7,
  };
  return weights[eventType] != null ? weights[eventType] : 0;
}

/**
 * Compute compounded WhatsApp risk score from WhatsAppRiskEvents with time decay.
 * Per-campaign cap: each campaign contributes at most 1.4 (one read + one click + one credentials).
 * Duplicate events (e.g. multiple reads/clicks) do not sum — we take max per type per campaign, then sum with decay.
 * This matches email semantics and prevents rawScore from blowing past 1.4 when the same user has many duplicate events.
 */
async function computeWhatsAppRiskScore(userId) {
  const id = userId && mongoose.Types.ObjectId.isValid(userId) ? new mongoose.Types.ObjectId(userId) : userId;
  const events = await WhatsAppRiskEvent.find({ userId: id }).sort({ createdAt: 1 }).lean();
  if (!events.length) {
    console.log("[WhatsAppRisk] computeWhatsAppRiskScore userId=", id?.toString(), "| no events, rawRisk=0");
    return 0;
  }

  const now = Date.now();
  // Group by campaign (null/undefined = one bucket for events without campaign)
  const byCampaign = new Map();
  for (const ev of events) {
    const key = ev.campaignId ? ev.campaignId.toString() : "__none__";
    if (!byCampaign.has(key)) byCampaign.set(key, { read: 0, click: 0, cred: 0, latestAt: 0 });
    const w = ev.weight != null ? ev.weight : getWhatsAppRiskWeight(ev.eventType);
    const createdAt = ev.createdAt ? new Date(ev.createdAt).getTime() : now;
    const bucket = byCampaign.get(key);
    bucket.latestAt = Math.max(bucket.latestAt, createdAt);
    if (ev.eventType === "whatsapp_read") bucket.read = Math.max(bucket.read, w);
    else if (ev.eventType === "whatsapp_clicked") bucket.click = Math.max(bucket.click, w);
    else if (ev.eventType === "whatsapp_credentials_submitted") bucket.cred = Math.max(bucket.cred, w);
  }

  let rawScore = 0;
  for (const [, bucket] of byCampaign) {
    const rawCampaign = Math.min(MAX_RAW_SCORE, bucket.read + bucket.click + bucket.cred);
    const daysSince = (now - bucket.latestAt) / (1000 * 60 * 60 * 24);
    const decay = Math.pow(WHATSAPP_RISK_DECAY_RATE, Math.max(0, daysSince));
    rawScore += rawCampaign * decay;
  }

  // Normalize by max possible raw (1.4), then clamp to [0, 1]
  const normalized = Math.min(1, Math.max(0, rawScore / MAX_RAW_SCORE));
  const score = Math.round(normalized * 100) / 100;
  console.log("[WhatsAppRisk] computeWhatsAppRiskScore userId=", id?.toString(), "| events=", events.length, "campaigns=", byCampaign.size, "rawScore=", rawScore.toFixed(4), "normalized=", score);
  return score;
}

/**
 * Update stored learning score (WhatsApp) on User model. Stored as 1 - risk so higher = better.
 * Only updates learningScoreWhatsapp — never whatsappRiskScore or any other field.
 */
async function updateUserWhatsAppRiskScore(userId) {
  console.log("[WhatsAppRisk] updateUserWhatsAppRiskScore called for userId=", userId?.toString());
  const user = await User.findById(userId).select("role " + USER_FIELD_LEARNING_SCORE_WHATSAPP).lean();
  if (!user) {
    console.log("[WhatsAppRisk] updateUserWhatsAppRiskScore skip: user not found", userId?.toString());
    return;
  }
  if (!isEligibleForWhatsAppRiskScoring(user.role)) {
    console.log("[WhatsAppRisk] updateUserWhatsAppRiskScore skip: role not eligible", { userId: userId?.toString(), role: user.role });
    return;
  }
  const previousStored = user[USER_FIELD_LEARNING_SCORE_WHATSAPP];
  const rawRisk = await computeWhatsAppRiskScore(userId);
  // No events = 1 (perfect). With events, learning score = 1 - risk (decreases on read/click/credentials). Same as email.
  const learningScore = rawRisk === 0 ? 1 : Math.round((1 - rawRisk) * 100) / 100;
  const valueToSet = Math.max(0, Math.min(1, learningScore));
  await User.updateOne(
    { _id: userId },
    { $set: { [USER_FIELD_LEARNING_SCORE_WHATSAPP]: valueToSet } }
  );
  await updateUserCombinedLearningScore(userId, { whatsapp: valueToSet }).catch((err) => console.error("[WhatsAppRisk] updateUserCombinedLearningScore failed:", err.message));
  console.log("[WhatsAppRisk] updateUserWhatsAppRiskScore done", { userId: userId.toString(), previousStored, rawRisk, learningScore, field: USER_FIELD_LEARNING_SCORE_WHATSAPP });
}

/**
 * Create a WhatsApp risk event and update the user's stored score.
 * Only records for eligible roles (affiliated, non_affiliated).
 */
async function recordWhatsAppRiskEvent(userId, eventType, campaignId = null, weight = null) {
  console.log("[WhatsAppRisk] recordWhatsAppRiskEvent called", { userId: userId?.toString(), eventType, campaignId: campaignId?.toString(), weight });
  if (!userId) {
    console.log("[WhatsAppRisk] Skip: no userId provided");
    return;
  }
  try {
    const user = await User.findById(userId).select("role").lean();
    if (!user) {
      console.log("[WhatsAppRisk] Skip: user not found", userId.toString());
      return;
    }
    if (!isEligibleForWhatsAppRiskScoring(user.role)) {
      console.log("[WhatsAppRisk] Skip: role not eligible for WhatsApp risk", { userId: userId.toString(), role: user.role });
      return;
    }
    const w = weight != null ? weight : getWhatsAppRiskWeight(eventType);
    await WhatsAppRiskEvent.create({
      userId,
      eventType,
      campaignId: campaignId || undefined,
      weight: w,
    });
    console.log("[WhatsAppRisk] WhatsAppRiskEvent created; updating user score for", userId.toString());
    await updateUserWhatsAppRiskScore(userId);
    console.log("[WhatsAppRisk] Recorded", eventType, "for userId", userId.toString());
  } catch (err) {
    console.error("[WhatsAppRisk] recordWhatsAppRiskEvent failed:", err.message, err.stack);
  }
}

module.exports = {
  isEligibleForWhatsAppRiskScoring,
  computeWhatsAppRiskScore,
  updateUserWhatsAppRiskScore,
  recordWhatsAppRiskEvent,
  getWhatsAppRiskWeight,
  WHATSAPP_RISK_DECAY_RATE,
  ELIGIBLE_ROLES,
  MAX_RAW_SCORE,
  USER_FIELD_LEARNING_SCORE_WHATSAPP,
};
