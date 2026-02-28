const mongoose = require("mongoose");
const Course = require("../models/Course");
const CourseProgress = require("../models/CourseProgress");
const User = require("../models/User");
const EmailTemplate = require("../models/EmailTemplate");
const WhatsAppTemplate = require("../models/WhatsAppTemplate");
const { isCourseCompleted, generateCertificateNumber } = require("./certificateController");
const Certificate = require("../models/Certificate");
const { getBadgeLabel } = require("../utils/badgeMapping");
const nodemailerService = require("../services/nodemailerService");
const { formatEmailForSending } = require("../services/emailFormatter");
const twilioService = require("../services/twilioService");
const Email = require("../models/Email");
const WhatsAppCampaign = require("../models/WhatsAppCampaign");

/** Max counts per course level: modules (total), sections per module, MCQ questions per module */
const COURSE_LIMITS = {
  basic: { maxModules: 5, maxSections: 5, maxMcq: 10 },
  advanced: { maxModules: 10, maxSections: 10, maxMcq: 15 },
};

/** Time limit (ms) to complete an activity after email is sent (e.g. open email and mark complete). Default 5 minutes. */
const ACTIVITY_TIME_LIMIT_MS = 5 * 60 * 1000;

function validateCourseLimits(level, modules) {
  const lim = COURSE_LIMITS[level] || COURSE_LIMITS.basic;
  const label = level === "advanced" ? "Advanced" : "Basic";
  if (!Array.isArray(modules)) return null;
  if (modules.length > lim.maxModules) {
    return `${label} course limit: max ${lim.maxModules} modules.`;
  }
  for (let i = 0; i < modules.length; i++) {
    const m = modules[i];
    const sections = Array.isArray(m.sections) ? m.sections : [];
    const quiz = Array.isArray(m.quiz) ? m.quiz : [];
    if (sections.length > lim.maxSections) {
      return `${label} course limit: max ${lim.maxSections} sections per module.`;
    }
    if (quiz.length > lim.maxMcq) {
      return `${label} course limit: max ${lim.maxMcq} MCQ questions per module.`;
    }
  }
  return null;
}

/**
 * GET /api/courses
 * List courses (all or filtered). Optionally by createdBy.
 * RBAC: Filters courses based on user role and organization.
 */
async function getCourses(req, res) {
  try {
    const user = req.user;
    if (!user) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }

    const { createdBy, sort = "newest", limit = "50", page = "1" } = req.query;
    const filter = {};
    if (createdBy) filter.createdBy = createdBy;

    // RBAC: Filter courses based on user role
    // System admins: See only courses with orgId = null (courses for non-affiliated users)
    // Client admins: See only courses from their organization
    // Affiliated users: See only courses from their organization
    // Non-affiliated users: See only courses with orgId = null (system admin courses)
    if (user.role === "system_admin") {
      // System admins see only courses for non-affiliated users (orgId = null)
      filter.orgId = null;
    } else if (user.role === "client_admin" || user.role === "affiliated") {
      // Client admins and affiliated users see only their organization's courses
      const userOrgId = user.orgId?._id?.toString() || user.orgId?.toString();
      if (userOrgId) {
        filter.orgId = userOrgId;
      } else {
        // User has no organization, return empty
        return res.status(200).json({
          success: true,
          courses: [],
          pagination: {
            page: 1,
            limit: 50,
            total: 0,
            pages: 1,
          },
        });
      }
    } else if (user.role === "non_affiliated") {
      // Non-affiliated users see only system admin courses (orgId = null)
      filter.orgId = null;
    } else {
      // Unknown role, return empty
      return res.status(200).json({
        success: true,
        courses: [],
        pagination: {
          page: 1,
          limit: 50,
          total: 0,
          pages: 1,
        },
      });
    }

    const sortOption = sort === "oldest" ? { createdAt: 1 } : { createdAt: -1 };
    const limitNum = Math.min(parseInt(limit, 10) || 50, 100);
    const pageNum = Math.max(1, parseInt(page, 10) || 1);
    const skip = (pageNum - 1) * limitNum;

    const [courses, total] = await Promise.all([
      Course.find(filter)
        .sort(sortOption)
        .skip(skip)
        .limit(limitNum)
        .lean(),
      Course.countDocuments(filter),
    ]);

    const needUserLookup = courses.filter(
      (c) => (c.createdByName == null || c.createdByName === "") && (c.createdByEmail == null || c.createdByEmail === "") && c.createdBy
    );
    const userIds = [...new Set(needUserLookup.map((c) => c.createdBy.toString()))];
    const users = userIds.length
      ? await User.find({ _id: { $in: userIds } }).select("displayName email").lean()
      : [];
    const userMap = Object.fromEntries(users.map((u) => [u._id.toString(), u]));

    const coursesWithCreator = courses.map((c) => {
      const hasStored = c.createdByName != null || c.createdByEmail != null;
      if (hasStored) {
        return {
          ...c,
          createdBy: { _id: c.createdBy, displayName: c.createdByName || "", email: c.createdByEmail || "" },
        };
      }
      const u = c.createdBy ? userMap[c.createdBy.toString()] : null;
      return {
        ...c,
        createdBy: u
          ? { _id: c.createdBy, displayName: u.displayName || "", email: u.email || "" }
          : { _id: c.createdBy, displayName: "", email: "" },
      };
    });

    return res.status(200).json({
      success: true,
      courses: coursesWithCreator,
      pagination: {
        page: pageNum,
        limit: limitNum,
        total,
        pages: Math.ceil(total / limitNum) || 1,
      },
    });
  } catch (error) {
    console.error("getCourses error:", error);
    return res.status(500).json({ success: false, error: "Failed to fetch courses" });
  }
}

/**
 * GET /api/courses/:id
 * Get a single course by id.
 * RBAC: Checks if user has access to this course.
 */
async function getCourseById(req, res) {
  try {
    const user = req.user;
    if (!user) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }

    const course = await Course.findById(req.params.id).lean();

    if (!course) {
      return res.status(404).json({ success: false, error: "Course not found" });
    }

    // RBAC: Check if user has access to this course
    const userOrgId = user.orgId?._id?.toString() || user.orgId?.toString();
    const courseOrgId = course.orgId?.toString() || null;

    if (user.role === "system_admin") {
      // System admins can only access courses for non-affiliated users (orgId = null)
      if (courseOrgId !== null) {
        return res.status(403).json({ success: false, error: "Access denied. System admins can only access courses for non-affiliated users" });
      }
    } else if (user.role === "client_admin" || user.role === "affiliated") {
      // Client admins and affiliated users can only access their organization's courses
      if (!userOrgId || userOrgId !== courseOrgId) {
        return res.status(403).json({ success: false, error: "Access denied to this course" });
      }
    } else if (user.role === "non_affiliated") {
      // Non-affiliated users can only access system admin courses (orgId = null)
      if (courseOrgId !== null) {
        return res.status(403).json({ success: false, error: "Access denied to this course" });
      }
    } else {
      return res.status(403).json({ success: false, error: "Access denied" });
    }

    const hasStored = course.createdByName != null || course.createdByEmail != null;
    let createdByPayload;
    if (hasStored) {
      createdByPayload = { _id: course.createdBy, displayName: course.createdByName || "", email: course.createdByEmail || "" };
    } else if (course.createdBy) {
      const u = await User.findById(course.createdBy).select("displayName email").lean();
      createdByPayload = u ? { _id: course.createdBy, displayName: u.displayName || "", email: u.email || "" } : { _id: course.createdBy, displayName: "", email: "" };
    } else {
      createdByPayload = { _id: null, displayName: "", email: "" };
    }

    const courseWithCreator = { ...course, createdBy: createdByPayload };

    return res.status(200).json({ success: true, course: courseWithCreator });
  } catch (error) {
    console.error("getCourseById error:", error);
    return res.status(500).json({ success: false, error: "Failed to fetch course" });
  }
}

/**
 * POST /api/courses
 * Create a new course. Body: { courseTitle, description, modules }.
 * createdBy set from req.user._id.
 * RBAC: Only system_admin and client_admin can create courses.
 * System admin courses have orgId = null (visible to non-affiliated users).
 * Client admin courses have orgId = their organization (visible to their org users).
 */
async function createCourse(req, res) {
  try {
    const user = req.user;
    if (!user || !user._id) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }

    // RBAC: Only system_admin and client_admin can create courses
    if (user.role !== "system_admin" && user.role !== "client_admin") {
      return res.status(403).json({ success: false, error: "Insufficient permissions to create courses" });
    }

    const userId = user._id;

    const { courseTitle, description, modules, badges, level } = req.body || {};
    if (!courseTitle || typeof courseTitle !== "string" || !courseTitle.trim()) {
      return res.status(400).json({
        success: false,
        error: "courseTitle is required",
      });
    }

    const normalizedModules = Array.isArray(modules)
      ? modules.map((m) => ({
          title: (m.title || "").trim(),
          sections: Array.isArray(m.sections)
            ? m.sections
                .map((s) => ({
                  title: (s.title || "").trim(),
                  material: (s.material || "").trim(),
                  urls: Array.isArray(s.urls) ? s.urls.map((u) => String(u).trim()).filter(Boolean) : [],
                  media: Array.isArray(s.media)
                    ? s.media
                        .map((m) => ({
                          type: m.type === 'video' ? 'video' : 'image',
                          url: String(m.url || "").trim(),
                          alt: String(m.alt || "").trim(),
                          caption: String(m.caption || "").trim(),
                          publicId: String(m.publicId || "").trim(),
                          subtitleUrl: String(m.subtitleUrl || "").trim(),
                          youtubeId: String(m.youtubeId || "").trim(), // YouTube video ID
                        }))
                        .filter((m) => m.url)
                    : [],
                }))
                .filter((s) => s.title || s.material || s.urls.length > 0 || s.media.length > 0)
            : [],
          quiz: Array.isArray(m.quiz)
            ? m.quiz
                .map((q) => ({
                  question: (q.question || "").trim(),
                  choices: Array.isArray(q.choices) ? q.choices.map((c) => String(c).trim()).filter(Boolean) : [],
                  correctIndex: Math.max(0, parseInt(q.correctIndex, 10) || 0),
                }))
                .filter((q) => q.question && q.choices.length > 0)
            : [],
          activityType: m.activityType === "email" || m.activityType === "whatsapp" ? m.activityType : null,
        }))
      : [];

    const userDoc = await User.findById(userId).select("displayName email").lean();
    const createdByName = (userDoc && userDoc.displayName) ? userDoc.displayName : "";
    const createdByEmail = (userDoc && userDoc.email) ? userDoc.email : "";

    const normalizedBadges = Array.isArray(badges)
      ? badges.map((b) => String(b).trim()).filter(Boolean)
      : [];

    const normalizedLevel =
      level === "advanced" || level === "basic" ? level : "basic";

    const limitError = validateCourseLimits(normalizedLevel, normalizedModules);
    if (limitError) {
      return res.status(400).json({ success: false, error: limitError });
    }

    // Set orgId based on creator role
    // System admins: orgId = null (visible to non-affiliated users)
    // Client admins: orgId = their organization (visible to their org users)
    let courseOrgId = null;
    if (user.role === "client_admin") {
      // Handle both populated and non-populated orgId
      const userOrgId = user.orgId?._id || user.orgId;
      if (!userOrgId) {
        return res.status(400).json({ success: false, error: "Client admin must belong to an organization" });
      }
      courseOrgId = userOrgId;
    } else if (user.role === "system_admin") {
      courseOrgId = null; // System admin courses are for non-affiliated users
    }

    const course = new Course({
      courseTitle: courseTitle.trim(),
      description: (description || "").trim(),
      level: normalizedLevel,
      modules: normalizedModules,
      createdBy: userId,
      createdByName,
      createdByEmail,
      badges: normalizedBadges,
      orgId: courseOrgId,
    });

    await course.save();
    const saved = await Course.findById(course._id).lean();
    const courseResponse = {
      ...saved,
      createdBy: { _id: saved.createdBy, displayName: saved.createdByName || "", email: saved.createdByEmail || "" },
    };

    return res.status(201).json({ success: true, course: courseResponse });
  } catch (error) {
    console.error("createCourse error:", error);
    return res.status(500).json({ success: false, error: "Failed to create course" });
  }
}

/**
 * PUT /api/courses/:id
 * Update a course. Body: { courseTitle, description, modules, badges }.
 * RBAC: Only system_admin and client_admin can update courses.
 * System admins can only update courses with orgId = null (for non-affiliated users).
 * Client admins can only update courses from their organization.
 */
async function updateCourse(req, res) {
  try {
    const user = req.user;
    if (!user || !user._id) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }

    // RBAC: Only system_admin and client_admin can update courses
    if (user.role !== "system_admin" && user.role !== "client_admin") {
      return res.status(403).json({ success: false, error: "Insufficient permissions to update courses" });
    }

    const courseId = req.params.id;
    if (!courseId || !mongoose.Types.ObjectId.isValid(courseId)) {
      return res.status(400).json({ success: false, error: "Invalid course id" });
    }
    const course = await Course.findById(courseId);
    if (!course) {
      return res.status(404).json({ success: false, error: "Course not found" });
    }

    // RBAC: System admins can only update courses for non-affiliated users (orgId = null)
    if (user.role === "system_admin") {
      const courseOrgId = course.orgId?.toString() || null;
      if (courseOrgId !== null) {
        return res.status(403).json({ success: false, error: "Access denied. System admins can only update courses for non-affiliated users" });
      }
    }
    // RBAC: Client admins can only update courses from their organization
    else if (user.role === "client_admin") {
      const userOrgId = user.orgId?._id?.toString() || user.orgId?.toString();
      const courseOrgId = course.orgId?.toString() || null;
      if (!userOrgId || userOrgId !== courseOrgId) {
        return res.status(403).json({ success: false, error: "Access denied. You can only update courses from your organization" });
      }
    }
    const { courseTitle, description, modules, badges, level } = req.body || {};
    if (courseTitle !== undefined) {
      if (typeof courseTitle !== "string" || !courseTitle.trim()) {
        return res.status(400).json({ success: false, error: "courseTitle must be a non-empty string" });
      }
      course.courseTitle = courseTitle.trim();
    }
    if (description !== undefined) course.description = (description || "").trim();
    if (level === "advanced" || level === "basic") course.level = level;
    if (Array.isArray(badges)) {
      course.badges = badges.map((b) => String(b).trim()).filter(Boolean);
    }
    if (Array.isArray(modules)) {
      const effectiveLevel = level === "advanced" || level === "basic" ? level : (course.level || "basic");
      const limitError = validateCourseLimits(effectiveLevel, modules);
      if (limitError) {
        return res.status(400).json({ success: false, error: limitError });
      }
      course.modules = modules.map((m) => ({
        title: (m.title || "").trim(),
        sections: Array.isArray(m.sections)
          ? m.sections
              .map((s) => ({
                title: (s.title || "").trim(),
                material: (s.material || "").trim(),
                urls: Array.isArray(s.urls) ? s.urls.map((u) => String(u).trim()).filter(Boolean) : [],
                media: Array.isArray(s.media)
                  ? s.media
                      .map((m) => ({
                        type: m.type === 'video' ? 'video' : 'image',
                        url: String(m.url || "").trim(),
                        alt: String(m.alt || "").trim(),
                        caption: String(m.caption || "").trim(),
                        publicId: String(m.publicId || "").trim(),
                        subtitleUrl: String(m.subtitleUrl || "").trim(),
                        youtubeId: String(m.youtubeId || "").trim(), // YouTube video ID
                      }))
                      .filter((m) => m.url)
                  : [],
              }))
              .filter((s) => s.title || s.material || (s.urls && s.urls.length > 0) || (s.media && s.media.length > 0))
          : [],
        quiz: Array.isArray(m.quiz)
          ? m.quiz
              .map((q) => ({
                question: (q.question || "").trim(),
                choices: Array.isArray(q.choices) ? q.choices.map((c) => String(c).trim()).filter(Boolean) : [],
                correctIndex: Math.max(0, parseInt(q.correctIndex, 10) || 0),
              }))
              .filter((q) => q.question && q.choices && q.choices.length > 0)
          : [],
        activityType: m.activityType === "email" || m.activityType === "whatsapp" ? m.activityType : null,
      }));
    }
    await course.save();
    const saved = await Course.findById(course._id).lean();
    const courseResponse = {
      ...saved,
      createdBy: {
        _id: saved.createdBy,
        displayName: saved.createdByName || "",
        email: saved.createdByEmail || "",
      },
    };
    return res.status(200).json({ success: true, course: courseResponse });
  } catch (error) {
    console.error("updateCourse error:", error);
    return res.status(500).json({ success: false, error: "Failed to update course" });
  }
}

/**
 * DELETE /api/courses/:id
 * Delete a course and all its progress records.
 * RBAC: Only system_admin and client_admin can delete courses.
 * System admins can only delete courses with orgId = null (for non-affiliated users).
 * Client admins can only delete courses from their organization.
 */
async function deleteCourse(req, res) {
  try {
    const user = req.user;
    if (!user || !user._id) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }

    // RBAC: Only system_admin and client_admin can delete courses
    if (user.role !== "system_admin" && user.role !== "client_admin") {
      return res.status(403).json({ success: false, error: "Insufficient permissions to delete courses" });
    }

    const courseId = req.params.id;
    if (!courseId || !mongoose.Types.ObjectId.isValid(courseId)) {
      return res.status(400).json({ success: false, error: "Invalid course id" });
    }
    const course = await Course.findById(courseId);
    if (!course) {
      return res.status(404).json({ success: false, error: "Course not found" });
    }

    // RBAC: System admins can only delete courses for non-affiliated users (orgId = null)
    if (user.role === "system_admin") {
      const courseOrgId = course.orgId?.toString() || null;
      if (courseOrgId !== null) {
        return res.status(403).json({ success: false, error: "Access denied. System admins can only delete courses for non-affiliated users" });
      }
    }
    // RBAC: Client admins can only delete courses from their organization
    else if (user.role === "client_admin") {
      const userOrgId = user.orgId?._id?.toString() || user.orgId?.toString();
      const courseOrgId = course.orgId?.toString() || null;
      if (!userOrgId || userOrgId !== courseOrgId) {
        return res.status(403).json({ success: false, error: "Access denied. You can only delete courses from your organization" });
      }
    }
    await CourseProgress.deleteMany({ course: courseId });
    await Course.findByIdAndDelete(courseId);
    return res.status(200).json({ success: true });
  } catch (error) {
    console.error("deleteCourse error:", error);
    return res.status(500).json({ success: false, error: "Failed to delete course" });
  }
}

/**
 * GET /api/courses/:courseId/progress
 * Get current user's progress for a course (completed submodule ids).
 */
async function getProgress(req, res) {
  try {
    const userId = req.user?._id;
    if (!userId) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }
    const { courseId } = req.params;
    const progress = await CourseProgress.findOne({ user: userId, course: courseId }).lean();
    const completed = progress?.completed ?? [];
    return res.status(200).json({ success: true, completed });
  } catch (error) {
    console.error("getProgress error:", error);
    return res.status(500).json({ success: false, error: "Failed to fetch progress" });
  }
}

/**
 * GET /api/courses/:courseId/progress/activity-email-status?submoduleId=0-activity
 * Get telemetry status for the activity email (opened, clicked, credentials). Used to show Pass/Fail on the submodule page.
 * passed = opened and not clicked and not credentials entered.
 */
async function getActivityEmailStatus(req, res) {
  try {
    const userId = req.user?._id;
    if (!userId) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }
    const { courseId } = req.params;
    const submoduleId = (req.query && req.query.submoduleId) || "";
    if (!submoduleId || !String(submoduleId).endsWith("-activity")) {
      return res.status(400).json({ success: false, error: "submoduleId (e.g. 0-activity) is required" });
    }
    const id = String(submoduleId).trim();
    const progress = await CourseProgress.findOne({ user: userId, course: courseId }).lean();
    const activityEmailIds = progress?.activityEmailIds;
    const emailId = activityEmailIds?.get?.(id) ?? activityEmailIds?.[id];
    if (!emailId) {
      return res.status(200).json({
        success: true,
        hasEmail: false,
        passed: null,
        openedAt: null,
        clickedAt: null,
        credentialsEnteredAt: null,
      });
    }
    const emailDoc = await Email.findById(emailId).select("openedAt clickedAt credentialsEnteredAt").lean();
    if (!emailDoc) {
      return res.status(200).json({
        success: true,
        hasEmail: true,
        passed: null,
        openedAt: null,
        clickedAt: null,
        credentialsEnteredAt: null,
      });
    }
    const opened = !!emailDoc.openedAt;
    const credentialsEntered = !!emailDoc.credentialsEnteredAt;
    // Only count click as fail if it happened AFTER open (avoids false positives from link prefetchers/scanners that hit the URL before user opens email)
    const clickedAfterOpen =
      emailDoc.openedAt &&
      emailDoc.clickedAt &&
      new Date(emailDoc.clickedAt) > new Date(emailDoc.openedAt);
    const passed = opened && !credentialsEntered && !clickedAfterOpen;
    return res.status(200).json({
      success: true,
      hasEmail: true,
      passed,
      openedAt: emailDoc.openedAt || null,
      clickedAt: emailDoc.clickedAt || null,
      credentialsEnteredAt: emailDoc.credentialsEnteredAt || null,
    });
  } catch (error) {
    console.error("getActivityEmailStatus error:", error);
    return res.status(500).json({ success: false, error: "Failed to fetch activity email status" });
  }
}

/**
 * POST /api/courses/:courseId/progress
 * Mark a submodule as complete. Body: { submoduleId } (e.g. "0-0", "0-quiz", "0-activity").
 * For email activity (N-activity): only allow completion if the sent email was opened and not clicked or credentials entered.
 */
async function markComplete(req, res) {
  try {
    const userId = req.user?._id;
    if (!userId) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }
    const { courseId } = req.params;
    const { submoduleId } = req.body || {};
    if (!submoduleId || typeof submoduleId !== "string" || !submoduleId.trim()) {
      return res.status(400).json({ success: false, error: "submoduleId is required" });
    }
    const id = submoduleId.trim();

    if (id.endsWith("-activity")) {
      const progress = await CourseProgress.findOne({ user: userId, course: courseId }).lean();
      const activityEmailIds = progress?.activityEmailIds;
      // activityEmailIds is a Map in schema but plain object when .lean(); support both
      const emailId =
        typeof activityEmailIds?.get === "function"
          ? activityEmailIds.get(id)
          : activityEmailIds?.[id];
      if (emailId) {
        const emailDoc = await Email.findById(emailId).lean();
        if (emailDoc) {
          const opened = !!emailDoc.openedAt;
          const credentialsEntered = !!emailDoc.credentialsEnteredAt;
          const clickedAfterOpen =
            emailDoc.openedAt &&
            emailDoc.clickedAt &&
            new Date(emailDoc.clickedAt) > new Date(emailDoc.openedAt);
          if (!opened) {
            return res.status(400).json({
              success: false,
              error: "Open the activity email first to pass. Do not click links or enter credentials.",
            });
          }
          if (clickedAfterOpen || credentialsEntered) {
            return res.status(400).json({
              success: false,
              error: "You clicked a link or entered credentials in the email. To pass this activity, open the email only—do not click links or enter credentials.",
            });
          }
          // Enforce 5-minute completion window from when the email was sent
          const sentAt = emailDoc.createdAt ? new Date(emailDoc.createdAt).getTime() : 0;
          if (sentAt && Date.now() - sentAt > ACTIVITY_TIME_LIMIT_MS) {
            return res.status(400).json({
              success: false,
              error: "Activity time limit exceeded. Complete within 5 minutes of receiving the email.",
            });
          }
        }
      }
    }

    const progress = await CourseProgress.findOneAndUpdate(
      { user: userId, course: courseId },
      { $addToSet: { completed: id } },
      { new: true, upsert: true }
    ).lean();

    // Check if course is now completed and generate certificate if needed
    const courseIsCompleted = await isCourseCompleted(userId, courseId);
    let certificateGenerated = false;
    if (courseIsCompleted) {
      // Check if certificate already exists
      const existingCert = await Certificate.findOne({ user: userId, course: courseId }).lean();
      if (!existingCert) {
        // Auto-generate certificate
        try {
          const course = await Course.findById(courseId).lean();
          const user = await User.findById(userId).select("displayName email").lean();
          if (course && user) {
            const certificate = new Certificate({
              user: userId,
              course: courseId,
              userName: user.displayName || "User",
              userEmail: user.email,
              courseTitle: course.courseTitle,
              courseDescription: course.description || "",
              certificateNumber: generateCertificateNumber(),
              issuedDate: new Date(),
              completionDate: new Date(),
            });
            await certificate.save();
            certificateGenerated = true;
          }
        } catch (certError) {
          console.error("Error auto-generating certificate:", certError);
          // Don't fail the request if certificate generation fails
        }
      }

      // Assign badges from course to user profile
      try {
        const course = await Course.findById(courseId).select("badges").lean();
        if (course && course.badges && Array.isArray(course.badges) && course.badges.length > 0) {
          // Transform badge IDs to labels before storing
          const badgeLabels = course.badges
            .map(badgeId => getBadgeLabel(badgeId))
            .filter(label => label !== null); // Filter out invalid badge IDs
          
          if (badgeLabels.length > 0) {
            // Add badge labels to user's profile (using $addToSet to avoid duplicates)
            await User.findByIdAndUpdate(
              userId,
              { $addToSet: { badges: { $each: badgeLabels } } },
              { new: true }
            );
          }
        }
      } catch (badgeError) {
        console.error("Error assigning badges:", badgeError);
        // Don't fail the request if badge assignment fails
      }
    }
    
    return res.status(200).json({
      success: true,
      completed: progress.completed,
      certificateGenerated
    });
  } catch (error) {
    console.error("markComplete error:", error);
    return res.status(500).json({ success: false, error: "Failed to update progress" });
  }
}

/**
 * DELETE /api/courses/:courseId/progress
 * Unmark a submodule as complete. Body: { submoduleId } (e.g. "0-0", "0-quiz").
 */
async function unmarkComplete(req, res) {
  try {
    const userId = req.user?._id;
    if (!userId) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }
    const { courseId } = req.params;
    const { submoduleId } = req.body || {};
    if (!submoduleId || typeof submoduleId !== "string" || !submoduleId.trim()) {
      return res.status(400).json({ success: false, error: "submoduleId is required" });
    }
    const id = submoduleId.trim();
    const progress = await CourseProgress.findOneAndUpdate(
      { user: userId, course: courseId },
      { $pull: { completed: id } },
      { new: true, upsert: true }
    ).lean();
    return res.status(200).json({ success: true, completed: progress.completed });
  } catch (error) {
    console.error("unmarkComplete error:", error);
    return res.status(500).json({ success: false, error: "Failed to update progress" });
  }
}

/** Title of the email template used for training course "email" activity (Dropbox phishing template) */
const ACTIVITY_EMAIL_TEMPLATE_TITLE = "Dropbox Shared Document";

/**
 * POST /api/courses/:courseId/activity/send-email
 * Send the Dropbox email template to the given address (training activity).
 * Body: { to: "user@example.com", submoduleId: "0-activity" } (submoduleId required to link email for pass/fail check).
 */
async function sendActivityEmail(req, res) {
  try {
    const userId = req.user?._id;
    if (!userId) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }
    const { courseId } = req.params;
    const { to, submoduleId } = req.body || {};

    if (!to || typeof to !== "string" || !to.trim()) {
      return res.status(400).json({ success: false, error: "Email address (to) is required" });
    }
    if (!submoduleId || typeof submoduleId !== "string" || !submoduleId.trim() || !submoduleId.endsWith("-activity")) {
      return res.status(400).json({ success: false, error: "submoduleId (e.g. 0-activity) is required" });
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(to.trim())) {
      return res.status(400).json({ success: false, error: "Invalid email address" });
    }

    const template = await EmailTemplate.findOne({
      title: ACTIVITY_EMAIL_TEMPLATE_TITLE,
      isActive: true,
    }).lean();

    if (!template || !template.emailTemplate) {
      return res.status(404).json({
        success: false,
        error: "Dropbox email template not found. Please seed email templates first.",
      });
    }

    const subject = template.emailTemplate.subject;
    const bodyContent = template.emailTemplate.bodyContent;
    const html = formatEmailForSending(bodyContent);

    const fromAddress = process.env.SMTP_FROM || process.env.SMTP_USER;
    if (!fromAddress) {
      return res.status(500).json({ success: false, error: "SMTP sender not configured" });
    }

    const emailRecord = new Email({
      sentBy: fromAddress,
      sentTo: to.trim(),
      subject,
      bodyContent,
      status: "pending",
    });
    await emailRecord.save();

    const result = await nodemailerService.sendEmail({
      to: to.trim(),
      subject,
      html,
      trackingEmailId: emailRecord._id,
    });

    emailRecord.messageId = result.success ? result.messageId : null;
    emailRecord.status = result.success ? "sent" : "failed";
    emailRecord.error = result.success ? null : (result.error || null);
    await emailRecord.save();

    if (result.success) {
      const sid = submoduleId.trim();
      await CourseProgress.findOneAndUpdate(
        { user: userId, course: courseId },
        { $set: { [`activityEmailIds.${sid}`]: emailRecord._id } },
        { upsert: true, new: true }
      );
    }

    if (!result.success) {
      return res.status(500).json({
        success: false,
        error: result.error || "Failed to send email",
      });
    }

    return res.status(200).json({
      success: true,
      message: "Email sent successfully",
      emailSentAt: emailRecord.createdAt,
      timeLimitMinutes: 5,
    });
  } catch (error) {
    console.error("sendActivityEmail error:", error);
    return res.status(500).json({ success: false, error: "Failed to send activity email" });
  }
}

/** Title of the WhatsApp template used for training course "whatsapp" activity (Dropbox) */
const ACTIVITY_WHATSAPP_TEMPLATE_TITLE = "Dropbox File Share";

/**
 * POST /api/courses/:courseId/activity/send-whatsapp
 * Send the Dropbox WhatsApp template to the given phone number (training activity).
 * Body: { to: "+1234567890" or "1234567890" }
 */
async function sendActivityWhatsApp(req, res) {
  try {
    const { courseId } = req.params;
    const { to } = req.body || {};

    if (!to || typeof to !== "string" || !to.trim()) {
      return res.status(400).json({ success: false, error: "Phone number (to) is required" });
    }

    const digitsOnly = to.trim().replace(/\D/g, "");
    if (digitsOnly.length < 10) {
      return res.status(400).json({ success: false, error: "Invalid phone number (at least 10 digits)" });
    }

    const template = await WhatsAppTemplate.findOne({
      title: ACTIVITY_WHATSAPP_TEMPLATE_TITLE,
      isActive: true,
    }).lean();

    if (!template || !template.messageTemplate) {
      return res.status(404).json({
        success: false,
        error: "Dropbox WhatsApp template not found. Please seed WhatsApp templates first.",
      });
    }

    const messageBody = template.messageTemplate;

    const result = await twilioService.sendWhatsAppMessage(to.trim(), messageBody);

    const userId = req.user?._id || null;
    const orgId = req.user?.orgId?._id || req.user?.orgId || null;
    const landingPageUrl = process.env.FRONTEND_URL || process.env.BACKEND_URL || "https://training-activity";
    const campaign = new WhatsAppCampaign({
      name: "Training Activity",
      description: "Training module WhatsApp activity",
      organizationId: orgId,
      createdBy: userId,
      managedByParentCampaign: false,
      templateId: "training_activity",
      targetUsers: [
        {
          phoneNumber: to.trim(),
          status: result.success ? "sent" : "failed",
          messageSid: result.success ? result.messageId : undefined,
          sentAt: result.success ? new Date() : undefined,
          failureReason: result.success ? undefined : (result.error || "Send failed"),
        },
      ],
      status: "completed",
      startDate: new Date(),
      endDate: new Date(),
      messageTemplate: messageBody,
      landingPageUrl,
      trackingEnabled: false,
      stats: {
        totalSent: result.success ? 1 : 0,
        totalDelivered: 0,
        totalRead: 0,
        totalClicked: 0,
        totalReported: 0,
        totalFailed: result.success ? 0 : 1,
      },
    });
    await campaign.save();

    if (!result.success) {
      return res.status(500).json({
        success: false,
        error: result.error || "Failed to send WhatsApp message",
      });
    }

    return res.status(200).json({ success: true, message: "WhatsApp message sent successfully" });
  } catch (error) {
    console.error("sendActivityWhatsApp error:", error);
    return res.status(500).json({ success: false, error: "Failed to send activity WhatsApp message" });
  }
}

module.exports = {
  getCourses,
  getCourseById,
  createCourse,
  updateCourse,
  deleteCourse,
  getProgress,
  getActivityEmailStatus,
  markComplete,
  unmarkComplete,
  sendActivityEmail,
  sendActivityWhatsApp,
};
