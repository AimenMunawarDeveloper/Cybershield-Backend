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
 * Mirrors email logic: raw = sum(weight * DECAY^daysSince). Normalized to [0, 1] by dividing by MAX_RAW_SCORE (1.4).
 */
async function computeWhatsAppRiskScore(userId) {
  const id = userId && mongoose.Types.ObjectId.isValid(userId) ? new mongoose.Types.ObjectId(userId) : userId;
  const events = await WhatsAppRiskEvent.find({ userId: id }).sort({ createdAt: 1 }).lean();
  if (!events.length) {
    console.log("[WhatsAppRisk] computeWhatsAppRiskScore userId=", id?.toString(), "| no events, rawRisk=0");
    return 0;
  }

  const now = Date.now();
  let rawScore = 0;
  for (const ev of events) {
    const createdAt = ev.createdAt ? new Date(ev.createdAt).getTime() : now;
    const daysSince = (now - createdAt) / (1000 * 60 * 60 * 24);
    const decay = Math.pow(WHATSAPP_RISK_DECAY_RATE, Math.max(0, daysSince));
    const weight = ev.weight != null ? ev.weight : getWhatsAppRiskWeight(ev.eventType);
    const contrib = weight * decay;
    rawScore += contrib;
    console.log("[WhatsAppRisk]   event", ev.eventType, "weight=", weight, "daysSince=", daysSince.toFixed(2), "decay=", decay.toFixed(4), "contrib=", contrib.toFixed(4));
  }

  // Normalize by max possible raw (1.4), then clamp to [0, 1] — same as email
  const normalized = Math.min(1, Math.max(0, rawScore / MAX_RAW_SCORE));
  const score = Math.round(normalized * 100) / 100;
  console.log("[WhatsAppRisk] computeWhatsAppRiskScore userId=", id?.toString(), "| events=", events.length, "rawScore=", rawScore.toFixed(4), "normalized=", score, "(raw/", MAX_RAW_SCORE, ")");
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
