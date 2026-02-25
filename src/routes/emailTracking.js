const express = require("express");
const { expressApp, getMailTrackingOptions } = require("../config/mailTracking");

// Log every request so we can see if tracking hits reach this server (debug: "no logs" = request never arrived)
const trackingApp = expressApp(() => getMailTrackingOptions());
const router = express.Router();
router.use((req, res, next) => {
  console.log("[Mail tracking] REQUEST HIT SERVER", { path: req.path, method: req.method, url: req.originalUrl });
  next();
});
router.use(trackingApp);

// Mounted at /api/track/email so full paths are /api/track/email/link/:jwt and /api/track/email/blank-image/:jwt
module.exports = router;
