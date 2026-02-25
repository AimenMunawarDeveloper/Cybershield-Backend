const mongoose = require("mongoose");

const emailSchema = new mongoose.Schema(
  {
    sentBy: {
      type: String,
      required: true,
      trim: true,
    },
    sentTo: {
      type: String,
      required: true,
      trim: true,
    },
    subject: {
      type: String,
      required: true,
    },
    bodyContent: {
      type: String,
      required: true,
    },
    messageId: {
      type: String,
      required: false,
    },
    status: {
      type: String,
      enum: ["sent", "failed"],
      default: "sent",
    },
    error: {
      type: String,
      required: false,
    },
    campaignId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Campaign",
      required: false,
    },
    // Telemetry (open = read, click) — set when tracking pixel loads / link is clicked
    openedAt: { type: Date, required: false, default: null },
    clickedAt: { type: Date, required: false, default: null },
  },
  {
    timestamps: true,
  }
);

// Indexes for better query performance
emailSchema.index({ sentBy: 1, createdAt: -1 });
emailSchema.index({ sentTo: 1, createdAt: -1 });

module.exports = mongoose.model("Email", emailSchema);

