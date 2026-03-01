const mongoose = require("mongoose");

const remedialAssignmentSchema = new mongoose.Schema(
  {
    user: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    course: { type: mongoose.Schema.Types.ObjectId, ref: "Course", required: true },
    reason: {
      type: String,
      enum: [
        "remedial_recognizing_risks",   // Recognizing Online Risks & Scams
        "remedial_advanced_phishing",    // Advanced Phishing Detection & Threat Analysis
        "remedial_advanced_defensive",  // Advanced Defensive Techniques & Email Security
        // Legacy (kept for existing documents)
        "email_activity",
        "whatsapp_activity",
        "overall_mid",
        "overall_low_basic",
        "email_low",
        "whatsapp_low",
        "overall_low_email",
        "overall_low_whatsapp",
      ],
      required: true,
    },
    assignedAt: { type: Date, default: Date.now },
    /** Deadline to complete the course (e.g. assignedAt + 30 days). */
    dueAt: { type: Date, required: false },
    /** Set when the user completes the course (all submodules done). */
    completedAt: { type: Date, required: false },
    /** Set when the user's score becomes high (67+) so remedial is no longer required; assignment is cancelled. */
    cancelledAt: { type: Date, required: false },
  },
  { timestamps: true }
);

remedialAssignmentSchema.index({ user: 1, reason: 1 }, { unique: true });
remedialAssignmentSchema.index({ user: 1 });

module.exports = mongoose.model("RemedialAssignment", remedialAssignmentSchema);
