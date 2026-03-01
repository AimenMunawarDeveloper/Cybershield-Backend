const Course = require("../models/Course");
const RemedialAssignment = require("../models/RemedialAssignment");
const User = require("../models/User");

const ELIGIBLE_ROLES = ["affiliated", "non_affiliated"];

// Score thresholds (same as UI: Low ≤0.33, Medium 0.34–0.66, High ≥0.67)
// For per-channel scores (0–1): low ≤ 0.33
const SCORE_LOW_CHANNEL = 0.33;
// For combined learningScore (0–100): low 0–33, mid 34–66
const SCORE_LOW_OVERALL_MAX = 33;
const SCORE_MID_OVERALL_MIN = 34;
const SCORE_MID_OVERALL_MAX = 66;

const COURSE_TITLE_PHISHING_BASIC = "Phishing basic";
const COURSE_TITLE_PHISHING_ADVANCE = "Phishing advance";

// Remedial course titles (by score rules)
const COURSE_RECOGNIZING_RISKS = "Recognizing Online Risks & Scams";
const COURSE_ADVANCED_PHISHING = "Advanced Phishing Detection & Threat Analysis";
const COURSE_ADVANCED_DEFENSIVE = "Advanced Defensive Techniques & Email Security";

/** (reason, courseTitle) for new remedial logic */
const REMEDIAL_REASON_TO_TITLE = {
  remedial_recognizing_risks: COURSE_RECOGNIZING_RISKS,
  remedial_advanced_phishing: COURSE_ADVANCED_PHISHING,
  remedial_advanced_defensive: COURSE_ADVANCED_DEFENSIVE,
};

/** Default days from assignment until deadline to complete the course. */
const REMEDIAL_DEADLINE_DAYS = 30;
const MS_PER_DAY = 24 * 60 * 60 * 1000;

function isEligibleForRemedial(role) {
  return role && ELIGIBLE_ROLES.includes(role);
}

/**
 * Get course filter for a user (same logic as getCourses RBAC / lmsRiskScoreService).
 */
function getCourseFilterForUser(user) {
  if (!user) return null;
  const filter = {};
  if (user.role === "system_admin") {
    filter.orgId = null;
  } else if (user.role === "client_admin" || user.role === "affiliated") {
    const orgId = user.orgId?._id?.toString() || user.orgId?.toString();
    if (!orgId) return null;
    filter.orgId = orgId;
  } else if (user.role === "non_affiliated") {
    filter.orgId = null;
  } else {
    return null;
  }
  return filter;
}

/**
 * Find one course that has at least one module with activityType "email".
 */
async function findOneCourseWithEmailActivity(filter) {
  const courses = await Course.find({
    ...filter,
    "modules.activityType": "email",
  })
    .select("_id courseTitle")
    .limit(1)
    .lean();
  return courses[0] || null;
}

/**
 * Find one course that has at least one module with activityType "whatsapp".
 */
async function findOneCourseWithWhatsAppActivity(filter) {
  const courses = await Course.find({
    ...filter,
    "modules.activityType": "whatsapp",
  })
    .select("_id courseTitle")
    .limit(1)
    .lean();
  return courses[0] || null;
}

/**
 * Find one course by title (case-insensitive match on courseTitle).
 */
async function findCourseByTitle(title, filter) {
  if (!title || typeof title !== "string" || !title.trim()) return null;
  const course = await Course.findOne({
    ...filter,
    courseTitle: new RegExp(`^${escapeRegex(title.trim())}$`, "i"),
  })
    .select("_id courseTitle")
    .lean();
  return course;
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Decide which remedial course titles to assign based on scores.
 * - Overall HIGH (67–100): no courses.
 * - Overall LOW (0–33) and email & whatsapp low: Recognizing Online Risks & Scams, Advanced Phishing Detection & Threat Analysis.
 * - Overall MID (34–66) and email & whatsapp low: Recognizing, Advanced Phishing, Advanced Defensive Techniques & Email Security.
 * - Overall MID and email & whatsapp high: only Advanced Defensive Techniques & Email Security.
 * - Overall MID and mixed (one low, one high): Recognizing, Advanced Phishing.
 */
function getDesiredRemedialReasons(overallScore, emailScore, whatsappScore) {
  const isOverallHigh = overallScore > SCORE_MID_OVERALL_MAX;
  const isOverallLow = overallScore <= SCORE_LOW_OVERALL_MAX;
  const isOverallMid = overallScore >= SCORE_MID_OVERALL_MIN && overallScore <= SCORE_MID_OVERALL_MAX;
  const emailLow = emailScore <= SCORE_LOW_CHANNEL;
  const whatsappLow = whatsappScore <= SCORE_LOW_CHANNEL;
  const emailAndWhatsappLow = emailLow && whatsappLow;
  const emailAndWhatsappHigh = !emailLow && !whatsappLow;

  if (isOverallHigh) return [];
  if (isOverallLow) {
    // When email and whatsapp learning score is low → Recognizing + Advanced Phishing
    return ["remedial_recognizing_risks", "remedial_advanced_phishing"];
  }
  if (isOverallMid) {
    if (emailAndWhatsappLow) {
      return ["remedial_recognizing_risks", "remedial_advanced_phishing", "remedial_advanced_defensive"];
    }
    if (emailAndWhatsappHigh) {
      return ["remedial_advanced_defensive"];
    }
    // Mixed: one low one high → Recognizing + Advanced Phishing
    return ["remedial_recognizing_risks", "remedial_advanced_phishing"];
  }
  return [];
}

/**
 * Ensure remedial assignments for a user based on learning scores.
 * Assigns specific courses by title:
 * - Low (email & whatsapp low): Recognizing Online Risks & Scams, Advanced Phishing Detection & Threat Analysis.
 * - Mid + email & whatsapp low: those two + Advanced Defensive Techniques & Email Security.
 * - Mid + email & whatsapp high: only Advanced Defensive Techniques & Email Security.
 * - High overall: no remedials.
 */
async function ensureRemedialAssignments(userId) {
  if (!userId) {
    console.log("[RemedialAssignment] ensureRemedialAssignments skip: no userId");
    return;
  }

  const user = await User.findById(userId)
    .select("role orgId learningScore learningScoreEmail learningScoreWhatsapp")
    .lean();
  if (!user) {
    console.log("[RemedialAssignment] ensureRemedialAssignments skip: user not found", userId.toString());
    return;
  }
  if (!isEligibleForRemedial(user.role)) {
    console.log("[RemedialAssignment] ensureRemedialAssignments skip: role not eligible", { userId: userId.toString(), role: user.role });
    return;
  }

  const filter = getCourseFilterForUser(user);
  if (!filter) {
    console.log("[RemedialAssignment] ensureRemedialAssignments skip: no course filter (check orgId)", { userId: userId.toString(), role: user.role, orgId: user.orgId?.toString?.() });
    return;
  }

  const emailScore = user.learningScoreEmail != null ? user.learningScoreEmail : 0;
  const whatsappScore = user.learningScoreWhatsapp != null ? user.learningScoreWhatsapp : 0;
  const overallScore = user.learningScore != null ? user.learningScore : 0; // 0–100
  console.log("[RemedialAssignment] ensureRemedialAssignments scores", { userId: userId.toString(), overallScore, emailScore, whatsappScore, role: user.role });

  const desiredReasons = getDesiredRemedialReasons(overallScore, emailScore, whatsappScore);

  // Overall HIGH → cancel all active assignments, assign nothing
  if (overallScore > SCORE_MID_OVERALL_MAX) {
    const result = await RemedialAssignment.updateMany(
      { user: userId, completedAt: { $exists: false }, cancelledAt: { $exists: false } },
      { $set: { cancelledAt: new Date() } }
    );
    if (result.modifiedCount > 0) {
      console.log("[RemedialAssignment] cancelled", result.modifiedCount, "for user", userId.toString(), "(score high)");
    }
    console.log("[RemedialAssignment] ensureRemedialAssignments skip: score high", { userId: userId.toString(), overallScore });
    return;
  }

  const now = new Date();
  const dueAt = new Date(now.getTime() + REMEDIAL_DEADLINE_DAYS * MS_PER_DAY);

  // Resolve course IDs for desired reasons (by title)
  const toCreate = [];
  for (const reason of desiredReasons) {
    const title = REMEDIAL_REASON_TO_TITLE[reason];
    if (!title) continue;
    const course = await findCourseByTitle(title, filter);
    if (course) {
      toCreate.push({ user: userId, course: course._id, reason, dueAt });
    } else {
      console.log("[RemedialAssignment] course not found for user", userId.toString(), "title", title, "filter", JSON.stringify(filter));
    }
  }

  const desiredCourseIds = new Set(toCreate.map((c) => c.course.toString()));

  // Cancel any active assignment whose course is not in the desired set (sync state)
  const active = await RemedialAssignment.find({
    user: userId,
    completedAt: { $exists: false },
    cancelledAt: { $exists: false },
  })
    .select("_id course reason")
    .lean();
  for (const a of active) {
    if (!a.course) continue;
    const courseIdStr = a.course.toString();
    if (!desiredCourseIds.has(courseIdStr)) {
      await RemedialAssignment.updateOne({ _id: a._id }, { $set: { cancelledAt: new Date() } });
      console.log("[RemedialAssignment] cancelled assignment (not in desired set)", { userId: userId.toString(), reason: a.reason });
    }
  }

  if (toCreate.length === 0) {
    console.log("[RemedialAssignment] ensureRemedialAssignments: no assignments to create", { userId: userId.toString(), desiredReasons });
    return;
  }

  try {
    const toInsert = [];
    for (const item of toCreate) {
      const updated = await RemedialAssignment.updateOne(
        { user: userId, reason: item.reason, cancelledAt: { $exists: true } },
        { $unset: { cancelledAt: 1 }, $set: { dueAt: item.dueAt } }
      );
      if (updated.modifiedCount === 0) {
        const exists = await RemedialAssignment.findOne({
          user: userId,
          reason: item.reason,
          completedAt: { $exists: false },
          cancelledAt: { $exists: false },
        }).select("_id").lean();
        if (!exists) toInsert.push(item);
      }
    }
    if (toInsert.length > 0) {
      await RemedialAssignment.insertMany(toInsert);
      console.log("[RemedialAssignment] created", toInsert.length, "for user", userId.toString(), "reasons:", toInsert.map((a) => a.reason));
    }
    if (toCreate.length > toInsert.length) {
      const reactivated = toCreate.length - toInsert.length;
      const alreadyHad = toCreate.length - reactivated;
      if (reactivated > 0) {
        console.log("[RemedialAssignment] reactivated", reactivated, "for user", userId.toString());
      }
    }
  } catch (err) {
    if (err.code !== 11000) {
      console.error("[RemedialAssignment] ensureRemedialAssignments failed:", err.message);
    }
  }
}

/**
 * Get remedial assignments for a user with course details (only incomplete, non-cancelled).
 * When user's overall score is HIGH (67+), returns empty so no remedial is shown.
 */
async function getRemedialAssignmentsForUser(userId) {
  const user = await User.findById(userId).select("learningScore").lean();
  if (!user) return [];
  const overallScore = user.learningScore != null ? user.learningScore : 0;
  if (overallScore > SCORE_MID_OVERALL_MAX) return []; // Score high (67+) → show nothing

  const assignments = await RemedialAssignment.find({
    user: userId,
    completedAt: { $exists: false },
    cancelledAt: { $exists: false },
  })
    .populate("course", "courseTitle description level modules")
    .sort({ assignedAt: 1 })
    .lean();
  const defaultDeadlineMs = REMEDIAL_DEADLINE_DAYS * MS_PER_DAY;
  return assignments.map((a) => {
    const assigned = a.assignedAt ? new Date(a.assignedAt) : new Date();
    const due = a.dueAt ? new Date(a.dueAt) : new Date(assigned.getTime() + defaultDeadlineMs);
    return { ...a, dueAt: due.toISOString() };
  });
}

/**
 * Mark all remedial assignments for this user + course as completed (called when course is finished).
 */
async function markRemedialAssignmentsCompletedForCourse(userId, courseId) {
  if (!userId || !courseId) return;
  const result = await RemedialAssignment.updateMany(
    { user: userId, course: courseId, completedAt: { $exists: false } },
    { $set: { completedAt: new Date() } }
  );
  if (result.modifiedCount > 0) {
    console.log("[RemedialAssignment] marked", result.modifiedCount, "completed for user", userId.toString(), "course", courseId.toString());
  }
}

module.exports = {
  isEligibleForRemedial,
  ensureRemedialAssignments,
  getRemedialAssignmentsForUser,
  markRemedialAssignmentsCompletedForCourse,
  findOneCourseWithEmailActivity,
  findOneCourseWithWhatsAppActivity,
  findCourseByTitle,
  getCourseFilterForUser,
  SCORE_LOW_CHANNEL,
  SCORE_LOW_OVERALL_MAX,
  SCORE_MID_OVERALL_MIN,
  SCORE_MID_OVERALL_MAX,
  COURSE_TITLE_PHISHING_BASIC,
  COURSE_TITLE_PHISHING_ADVANCE,
  COURSE_RECOGNIZING_RISKS,
  COURSE_ADVANCED_PHISHING,
  COURSE_ADVANCED_DEFENSIVE,
  getDesiredRemedialReasons,
  REMEDIAL_DEADLINE_DAYS,
};
