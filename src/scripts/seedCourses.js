/**
 * Seed CyberShield LMS courses. Run: node src/scripts/seedCourses.js
 * Full content per section; max 5 sections per module (basic), 10 (advanced).
 */
const mongoose = require("mongoose");
const Course = require("../models/Course");
const User = require("../models/User");
require("dotenv").config();

function section(title, material) {
  return { title, material: material || "", urls: [], media: [] };
}
function mcq(question, choices, correctIndex = 0) {
  return { question, choices, correctIndex };
}
function mod(title, sections, quiz = [], activityType = null) {
  return { title, sections, quiz, activityType };
}

// ========== BASIC COURSE 1: Digital Safety & Personal Cyber Hygiene ==========
// Module 1: max 5 sections with full content
const M1_S1 = `Cybersecurity is the practice of protecting: your personal information, your online accounts, your devices, your money, and your identity from being stolen, misused, or damaged.

In simple words: Cybersecurity means staying safe online. Just like we lock our house doors to prevent theft, we must protect our digital "doors" — passwords, devices, and accounts.

Examples of what cybersecurity protects: Gmail (personal emails and recovery data), WhatsApp (private conversations), Instagram (photos and personal identity), Bank App (financial savings), University Portal (academic records).`;

const M1_S2 = `Today, most people in Pakistan use: smartphones, WhatsApp, online banking, social media, e-commerce platforms, and online learning systems. This means: our lives are connected to the internet, our money is connected to the internet, our identity is connected to the internet. If something goes wrong online, the consequences are real.

Cybercrime is increasing globally and locally. In Pakistan, common issues include: fake prize messages, bank impersonation scams, job offer scams, scholarship fraud, fake loan approvals, and social media account hacking. Because WhatsApp and mobile banking are widely used, scammers often target students, parents, small business owners, and university staff. Digital safety is no longer optional — it is necessary.`;

const M1_S3 = `Cybercrime does not only affect money. It affects: emotional well-being, academic reputation, family trust, and mental health.

Scenario 1 – Student: A university student receives "You have been selected for an international scholarship. Submit your documents and pay a small processing fee." The student sends money. Result: financial loss, emotional stress, embarrassment, loss of trust.

Scenario 2 – Parent: A parent receives "Your bank account will be blocked. Share OTP immediately." They share the OTP. Result: bank account emptied, savings lost.

Scenario 3 – Social media: A teenager accepts a friend request from a fake profile, shares personal photos. Later: blackmail attempt, emotional trauma.

Important reality: Cybercrime often exploits trust, uses fear, creates urgency, and targets emotions. It does NOT require advanced hacking skills. It targets human behavior. Technology alone cannot protect you. Even the strongest systems fail if a person shares OTP, uses weak passwords, clicks unknown links, or trusts unknown callers. Cybersecurity starts with awareness and smart decisions.`;

const M1_S4 = `Digital hygiene means forming safe habits.

Basic Digital Safety Rules: 1) Think before clicking. 2) Never share OTP with anyone. 3) Do not trust unknown callers. 4) Use strong passwords. 5) Verify information before sending money. 6) Avoid oversharing personal information. 7) Do not panic under pressure.

Pause Before Action Rule: If a message creates fear, urgency, excitement, or pressure — pause. Scammers want you to react emotionally. Safe users respond logically.

Digital Footprint: Every action online leaves a trace (posts, comments, likes, uploaded photos, shared documents). Employers, universities, and even scammers can analyze digital footprints. Think before posting: Would I be comfortable if this became public? Could this information be misused?

Cyber hygiene means keeping your digital environment clean and secure: updating software, using secure passwords, reviewing privacy settings, avoiding unsafe downloads, being cautious online. Just like brushing your teeth daily prevents dental issues, small digital habits prevent major security issues.`;

const M1_S5 = `Cybersecurity Myths:

Myth 1: "I am not important. No one will hack me." Reality: Scammers target everyone.

Myth 2: "Only rich people get targeted." Reality: Anyone with a bank account or phone can be targeted.

Myth 3: "If it looks professional, it must be real." Reality: Scammers copy official logos and designs.

Myth 4: "I will recognize a scam immediately." Reality: Many scams look very convincing.

Summary: Cybersecurity is not technical — it is behavioral. Staying safe online means thinking carefully, acting responsibly, protecting personal information, and avoiding emotional reactions. Digital safety protects not just you — but your family, finances, and future.`;

// Module 2: Passwords & Account Protection – 5 sections full content
const M2_S1 = `A password is the key to your digital identity. If someone gets access to your Gmail they can reset all other accounts; your WhatsApp they can impersonate you; your Instagram they can scam your friends; your banking app they can steal your money. Most cyber incidents begin with weak passwords, shared passwords, or reused passwords.

Real-Life Scenario: Ali uses the same password for Gmail, Instagram, Daraz, and university portal. Daraz gets breached. Hackers try the same email + password combination on Gmail. It works. Result: Gmail hacked, password reset links sent, other accounts compromised. This is called Credential Stuffing.`;

const M2_S2 = `Weak passwords include: your name + 123, birthdate, phone number, "password", "admin", simple patterns like 123456, repeated characters. Weak examples: Ali123, 03001234567, Pakistan1, qwerty, abc123. These are easily guessed within seconds by automated tools.

A strong password: at least 12 characters long, uses uppercase and lowercase letters, includes numbers and symbols, and is not based on personal information. Strong example: Blue!River#82Mango. It is hard to guess because it is long, random, mixes characters, and contains no personal information.

Passphrases: Instead of complicated symbols, you can use passphrases. Example: MyDogRunsFastInSummer! Passphrases are easier to remember, harder to guess, and longer (which increases security). Length increases strength more than complexity.`;

const M2_S3 = `Many people reuse passwords because it is easier and they do not want to remember multiple passwords. But if one site gets hacked, all accounts become vulnerable. When websites are hacked, email addresses and passwords are leaked; hackers test those credentials on other platforms. If you reuse passwords you are at high risk.

Password managers: store all your passwords securely, generate strong passwords, encrypt your credentials, and automatically fill login details. Examples: Google Password Manager, Bitwarden, 1Password, Apple iCloud Keychain. Instead of remembering 20 passwords you remember only one master password. The manager creates long random passwords, prevents reuse, and alerts you about compromised accounts.`;

const M2_S4 = `Even strong passwords can be stolen. That is why 2FA adds a second layer. Login = Password + Something else. Types of 2FA: OTP via SMS, Authenticator App (e.g. Google Authenticator), Fingerprint or Face ID, Hardware security key. Even if someone knows your password they cannot log in without the second factor. This protects banking apps, Gmail, social media, and university portals.

Never Share OTP: OTP is a One-Time Password. If someone asks "Share OTP to verify your account" it is a scam. Banks NEVER ask for OTP. OTP is your digital signature. Never share it — not with bank callers, delivery agents, friends, family, or anyone.`;

const M2_S5 = `Protecting Social Media Accounts: Make profile private, remove public phone number, turn off public tagging, enable login alerts, enable 2FA. If hacked, scammer may message your friends asking for money, share fake investment links, or cause reputation damage.

Protecting Financial Accounts: For banking apps, Easypaisa, JazzCash, online shopping — use unique password, enable 2FA, do not save banking passwords on shared devices, avoid logging in on public WiFi.

Recognizing Suspicious Login Activity: Warning signs — email "New login detected from unknown location", password reset you did not request, device logged out automatically. Action steps: change password immediately, enable 2FA, check recovery email, remove unknown devices. Password mistakes to avoid: writing passwords on paper near laptop, saving passwords in WhatsApp chat, sharing Netflix/Spotify passwords publicly, using same password everywhere, ignoring security alerts.`;

// Module 3: Safe Social Media & Messaging Use – 5 sections full content
const M3_S1 = `Social media and messaging apps are part of everyday life. We use them to communicate with friends and family, share photos and memories, follow news and trends, and conduct business and academic discussions. However, social media platforms are also common targets for scams, impersonation, harassment, identity theft, and blackmail.

Most students and families in Pakistan use WhatsApp, Instagram, Facebook, TikTok, Snapchat, LinkedIn. Social media connects us but also exposes information about where we live, where we study, who our friends are, what we like, and when we travel. This information can be misused. Privacy settings control: who sees your posts, who can message you, who can view your phone number, who can see your friends list, who can tag you. Safe practices: set account to "Private" when possible, limit post visibility to "Friends Only", review tagged photos before they appear, hide phone number from public view, disable location sharing in posts. Unsafe practices: public profiles without restriction, sharing phone number publicly, allowing anyone to message you, accepting all friend requests.`;

const M3_S2 = `One of the most common online risks is impersonation. Scammers create fake profiles pretending to be attractive strangers, university staff, bank representatives, job recruiters, celebrities, or influencers.

How to identify fake profiles: very few photos, recently created account, no mutual friends, poor grammar in messages, asking for money quickly, asking to move conversation off-platform. Student example: A fake "HR recruiter" contacts a student on Instagram: "We saw your profile. You are selected for an international internship. Pay Rs. 5000 processing fee." Red flags: no official email domain, no company website, payment request immediately.`;

const M3_S3 = `Oversharing means posting too much personal information. Examples: posting boarding passes, exam roll numbers, CNIC, sharing home address, live location tagging continuously, posting expensive purchases publicly. Why oversharing is dangerous: it can lead to identity theft, targeted scams, stalking, blackmail, and social engineering attacks. Before posting, ask: Would I share this with a stranger? Could this be misused? Is this information sensitive? If unsure — do not post.

Location privacy: Many apps automatically attach location to photos, stories, and check-ins. Public location sharing can expose your daily routine, your home location, and when your house is empty. Safe location practices: turn off automatic location tagging, avoid live public check-ins, share travel photos after returning, only share location with trusted contacts.`;

const M3_S4 = `Messaging apps like WhatsApp are commonly used for scams. Never: share OTP via WhatsApp, share bank screenshots, share personal documents, or click unknown shortened links. Common messaging scams: "Send OTP to verify your account", "Your bank account will be blocked", "You won a prize", "Your relative is in emergency", "Forward this message to 10 people".

Emotional manipulation: Scammers use fear ("Your account will be deleted"), excitement ("You won Rs. 1,000,000"), sympathy ("I need urgent help"), authority ("I am from FIA"). Emotional reactions reduce logical thinking. Pause before responding. Photos can be misused for fake profiles, editing and misuse, blackmail, and deepfake content. Safe photo sharing: avoid sending private photos, do not trust unknown online relationships, report blackmail immediately, do not respond to threats. Not everyone online is who they claim to be. If someone quickly becomes overly friendly, requests personal photos, asks to move to private chat, or requests money — it is a red flag.`;

const M3_S5 = `All platforms allow blocking users, reporting fake accounts, and reporting harassment. Blocking is not rude; it is protective. Your online presence affects university admissions, job opportunities, professional image, and personal relationships. Employers sometimes review public posts, comments, and online behavior. Think long-term before posting.

Safe Social Media Checklist: Is my account private? Have I enabled 2FA? Do I avoid sharing sensitive information? Do I accept friend requests carefully? Do I avoid sharing OTP? Do I review privacy settings regularly?`;

// Module 4: Device & Internet Safety – 5 sections full content
const M4_S1 = `Your smartphone, laptop, and internet connection are gateways to your digital life. Your device contains photos, messages, banking apps, university accounts, saved passwords, and contacts. If someone gains access to your device they gain access to your digital life. Protecting your device is protecting yourself.

Public WiFi is available in cafes, airports, shopping malls, universities, and hotels. But public WiFi is often not encrypted, easy to monitor, and vulnerable to attackers. An attacker on the same network may: monitor traffic, capture login credentials, create fake WiFi hotspots, or inject malicious links. Scenario: You connect to WiFi named "Free_Cafe_WiFi" — it is actually a fake hotspot. You log into Instagram and your credentials are captured. Safe public WiFi practices: avoid online banking on public WiFi, avoid entering sensitive passwords, use mobile data for financial transactions, turn off auto-connect WiFi, forget networks after use.`;

const M4_S2 = `When your phone or laptop says "Software update available" it is often fixing security weaknesses, bugs, and vulnerabilities. Hackers often exploit known weaknesses. When companies discover security flaws they release updates to fix them. If you ignore updates your device remains vulnerable. Common mistake: "I will update later." Delaying updates increases risk. Safe update habits: enable automatic updates, update apps regularly, update operating system, remove unused apps.`;

const M4_S3 = `Malware often spreads through fake apps, cracked software, pirated movies, and unofficial APK files. Dangerous sources: random websites, Telegram file links, pirated software sites, email attachments from unknown senders. Safe download sources: Google Play Store, Apple App Store, official company websites, verified university portals. Before installing an app check: number of downloads, reviews, developer name, requested permissions. Warning signs: app asks for unnecessary permissions (e.g. flashlight app asking for contacts access), no reviews, recently published, poor grammar in description, fake company name. Apps sometimes request camera, microphone, contacts, location, storage — ask: does this app really need this permission? A calculator app does NOT need microphone access.`;

const M4_S4 = `When browsing: check website spelling, look for HTTPS (lock icon), avoid clicking pop-up ads, do not download browser extensions randomly. Pop-up scam example: "Your phone is infected with 5 viruses! Click to clean." This is usually fake. Never click panic pop-ups. Avoid plugging unknown USB drives, using random charging cables, or using public charging stations without caution. Some attackers use USB malware or data theft cables. Home WiFi: change default router password, use strong WiFi password, avoid sharing password publicly, hide SSID if possible. Lock screen: always use PIN or password, enable fingerprint or Face ID, auto-lock after short inactivity. If phone is stolen and unlocked everything is accessible.`;

const M4_S5 = `If device is lost or hacked you can recover data if you use cloud backup, enable automatic backup, and sync contacts to Google or iCloud. Backup protects against device theft, accidental deletion, and ransomware. Signs your device may be compromised: battery drains quickly, phone overheats, unknown apps installed, pop-ups appear frequently, slow performance suddenly, login alerts from unknown locations. If suspected: disconnect from internet, remove suspicious apps, change passwords, scan device, seek technical help. Device safety checklist: Is my device updated? Do I avoid pirated apps? Do I use lock screen protection? Do I avoid public WiFi for banking? Do I check app permissions? Do I have backup enabled?`;

const BASIC_1 = {
  courseTitle: "Digital Safety & Personal Cyber Hygiene",
  description: "Introduction to cybersecurity in simple, practical terms: digital world, passwords and account protection, safe social media and messaging use, device and internet safety. Designed for learners in Pakistan.",
  level: "basic",
  modules: [
    mod("Understanding the Digital World", [
      section("What is Cybersecurity?", M1_S1),
      section("The Digital World & Why Digital Safety Matters in Pakistan", M1_S2),
      section("How Cybercrime Affects People & The Human Factor", M1_S3),
      section("Safe Online Behavior, Digital Footprint & Cyber Hygiene", M1_S4),
      section("Cybersecurity Myths & Summary", M1_S5),
    ], [
      mcq("What does cybersecurity primarily protect?", ["Only money", "Personal information, accounts, devices, money, and identity", "Only social media"], 1),
      mcq("Why is OTP sharing dangerous?", ["It is not", "OTP is like your digital signature; anyone with it can access or steal from your account", "Only banks need OTP"], 1),
      mcq("What is a digital footprint?", ["A type of password", "The trace of your actions online (posts, comments, likes, uploads)", "A security tool"], 1),
    ]),
    mod("Passwords & Account Protection", [
      section("Why Passwords Matter & Credential Stuffing", M2_S1),
      section("Strong vs Weak Passwords & Passphrases", M2_S2),
      section("Password Reuse & Password Managers", M2_S3),
      section("Two-Factor Authentication & Never Share OTP", M2_S4),
      section("Protecting Social & Financial Accounts", M2_S5),
    ], [
      mcq("Why is password reuse dangerous?", ["It is not", "If one site is breached, attackers try the same credentials elsewhere", "Passwords expire"], 1),
      mcq("Should you ever share OTP?", ["Yes if the caller says they are from the bank", "No — legitimate organizations never ask for OTP", "Only with family"], 1),
    ]),
    mod("Safe Social Media & Messaging Use", [
      section("Privacy Settings & The Role of Social Media", M3_S1),
      section("Fake Profiles & Impersonation", M3_S2),
      section("Oversharing & Location Privacy", M3_S3),
      section("Safe Messaging, Emotional Manipulation & Photos", M3_S4),
      section("Blocking, Reporting & Reputation Checklist", M3_S5),
    ], [
      mcq("What is oversharing?", ["Posting often", "Sharing too much personal information online", "Having a public profile"], 1),
      mcq("Should you share OTP via WhatsApp?", ["Yes if asked by support", "No — never share OTP with anyone", "Only for banking"], 1),
    ]),
    mod("Device & Internet Safety", [
      section("Your Device as Digital Identity & Public WiFi Risks", M4_S1),
      section("Software Updates & Security Patches", M4_S2),
      section("Safe Downloads & Recognizing Unsafe Apps", M4_S3),
      section("Safe Browsing, USB, Home WiFi & Lock Screen", M4_S4),
      section("Backup, Signs of Compromise & Device Checklist", M4_S5),
    ], [
      mcq("Why is public WiFi risky?", ["It is not", "Often unencrypted; attackers can monitor traffic or set up fake hotspots", "It is slow"], 1),
      mcq("Why are software updates important?", ["They are not", "They often fix security vulnerabilities", "Only for new features"], 1),
    ]),
  ],
};

// ========== BASIC COURSE 2: Recognizing Online Risks & Scams ==========
// 4 modules, 5 sections each max, full content. Last module has WhatsApp activity.
const R1_S1 = `An online scam is when someone uses the internet, messaging apps, or phone calls to trick you, manipulate you, steal money, or steal information. Scammers often pretend to be banks, government officials, university staff, employers, delivery services, or family members. They rely on trust and urgency.`;

const R1_S2 = `Fake Prize & Lottery Scams: You receive "Congratulations! You have won Rs. 500,000. Send processing fee to claim." Red flags: you never entered a contest, they ask for a "small fee", they request bank details, they create urgency. Student scenario: "You won iPhone 15. Pay Rs. 2,000 delivery charge." The student pays; phone never arrives. Golden rule: If you did not participate, you cannot win.

Fake Job Offer Scams: "Work from home. Earn Rs. 50,000 weekly. No interview required." Red flags: no official company email, Gmail or Yahoo address, no interview process, asking for registration fee or CNIC copy immediately. In Pakistan common fake job scams include data entry jobs, Amazon reseller jobs, overseas employment offers, fake government hiring. Real employers conduct interviews, use official company domains, and do NOT ask for payment.`;

const R1_S3 = `Bank Impersonation Scams: Very common in Pakistan. "Your account will be blocked. Send OTP immediately." Or "I am calling from your bank security department." Red flags: urgent tone, threat of account blocking, asking for OTP, PIN, or full card number. Banks NEVER ask for OTP, PIN, or full card number. If someone asks — it is a scam.

Loan & Investment Scams: "Instant loan approved in 5 minutes." Or "Invest Rs. 10,000 and earn Rs. 1,00,000." Red flags: unrealistic profits, no official website, pressure to invest quickly, WhatsApp-only communication. If returns sound too good to be true they are likely fake.`;

const R1_S4 = `Government & Authority Impersonation: Scammers pretend to be FIA officers, police, tax officials, NADRA, or university administration. "You are under investigation. Pay fine immediately." Red flags: threatening tone, demanding immediate payment, asking for money via transfer, refusing to provide official documentation. Real authorities do not demand payment via WhatsApp, do not ask for OTP, and follow official procedures. Delivery & Parcel Scams: "Your parcel is stuck at customs. Pay Rs. 3,000 to release." Red flags: you did not order anything, strange website link, payment via mobile wallet only. Family Emergency Scams: "I lost my phone. This is my new number. Send money urgently." Or "Your child is in trouble. Send money now." Scammers exploit fear. Safe action: always verify by calling the person directly on their known number.`;

const R1_S5 = `Emotional Manipulation Tactics: Scammers use fear ("Your account will be blocked"), excitement ("You won a prize!"), sympathy ("I need urgent help"), authority ("I am from the government"). Emotions override logic. Pause before acting. Why smart people still fall for scams: people are busy, people panic, messages look official, scammers sound confident. Awareness reduces vulnerability. Scam recognition checklist: Did I initiate this? Are they asking for money? Are they asking for OTP? Are they creating urgency? Does this sound too good to be true? If YES — pause. Safe habits: never share OTP, never pay random fees, verify through official sources, do not react emotionally.`;

const R2_S1 = `Suspicious messages may come through SMS, WhatsApp, email, Instagram DM, Facebook Messenger, Telegram, or phone calls. The platform does not matter; the patterns matter. Some scam messages look professional, use official logos, sound urgent, use correct grammar, and appear to come from trusted sources. The goal is to notice patterns before reacting.`;

const R2_S2 = `Warning Sign #1 – Urgency: Scammers often create panic. Examples: "Final warning!", "Act now or your account will be closed!", "Respond within 10 minutes!", "Immediate action required!" Urgency reduces thinking time. When people panic they react emotionally, skip verification, and ignore red flags. Safe response: pause, take a deep breath, verify before responding.

Warning Sign #2 – Requests for OTP or Password: If any message says "Send OTP to confirm", "Share your PIN for verification", or "Forward the code you received" — it is a scam. No legitimate organization will ask for OTP, password, ATM PIN, or full debit/credit card number. Never share them.`;

const R2_S3 = `Warning Sign #3 – Suspicious Links: "Click here to secure your account." Red flags: shortened links, strange spelling in website name (e.g. paypaI.com with capital I instead of L, hbl-support-help.xyz), random numbers in link, domain that looks slightly different. Even small differences matter. At basic level do not analyze technically; just avoid clicking unknown links.

Warning Sign #4 – Unusual Requests: "Send money immediately.", "Send your CNIC.", "Send bank screenshot.", "Transfer small fee to unlock reward." Ask yourself: why are they asking for this?`;

const R2_S4 = `Warning Sign #5 – Unknown Numbers Claiming Authority: "I am calling from FIA.", "This is your bank security department.", "I am from your university administration." Red flags: unknown mobile number, refusal to provide official contact, demanding immediate action. Always verify independently.

Warning Sign #6 – Too Good to Be True: "Earn Rs. 50,000 per week without work.", "Guaranteed crypto profit.", "Scholarship without application.", "Free iPhone." If it sounds unrealistic it probably is.`;

const R2_S5 = `The Pause Rule: Whenever you see suspicious signs — STOP. Do not reply immediately. Scammers rely on speed; you should rely on logic. Verification basics: instead of replying to the message, call official bank helpline, visit official website directly, contact university through official email, or call family member on known number. Never verify using the contact provided in the suspicious message. Common psychological tricks: fear ("You are under investigation"), authority ("I am from government"), excitement ("You won a prize!"), scarcity ("Offer expires in 10 minutes"), sympathy ("I need urgent help"). Recognizing the emotion helps you stay calm. Checklist before responding: Did I expect this message? Are they asking for sensitive information? Are they creating urgency? Are they threatening consequences? Does the link look strange? Does this sound too good to be true? If YES to any — do not proceed.`;

const R3_S1 = `Scammers succeed not because people are unaware but because people react quickly. They rely on panic, excitement, fear, and urgency. Your biggest protection is not technology; it is time. Pausing breaks the scam. STEP 1: STOP. When you receive a suspicious message do nothing immediately. Do not reply, click, call back, send money, or share OTP. Even 30 seconds of pause reduces risk significantly. Example: Message "Your bank account will be blocked in 5 minutes!" Correct first action: STOP. Do not respond.`;

const R3_S2 = `STEP 2: THINK. Ask yourself: Did I expect this message? Does this make logical sense? Why are they creating urgency? Why are they asking for sensitive information? Scammers want emotion; you use logic. Example: Message "You won Rs. 1,000,000." Think: Did I enter any contest? Why would they ask for a processing fee? Why are they messaging randomly? If logic fails — it is likely fake.`;

const R3_S3 = `STEP 3: VERIFY. Never verify using the contact information provided in the suspicious message. Instead: call official bank helpline from official website, visit official university website, call family member on known saved number, check company's verified social media page. Unsafe verification: calling the number provided in the suspicious message. Scammers often control that number. Example: Message "Your university portal is suspended. Click link." Safe action: open official university website directly, log in normally, do not click the link in the message.`;

const R3_S4 = `STEP 4: ACT. After verifying: if it is fake — delete message, block sender, report if necessary. If it is real — follow official instructions carefully. Never act before verifying. Applying the framework: Scenario 1 Fake Bank Call — "I am from your bank security team. Share OTP immediately." STOP (do not share), THINK (banks never ask OTP), VERIFY (call official bank helpline), ACT (block number). Scenario 2 Fake Job Offer — "Earn Rs. 50,000 weekly. Pay Rs. 3,000 registration." STOP (do not pay), THINK (why payment before job?), VERIFY (search official company website), ACT (ignore and block). Scenario 3 Emergency Family Message — "I am your cousin. New number. Send money urgently." STOP, THINK (why new number?), VERIFY (call cousin's old number), ACT (confirm before anything).`;

const R3_S5 = `Controlling emotional reactions: Scammers trigger fear, greed, excitement, sympathy, and authority pressure. Emotional control reduces risk. Take a deep breath; step away from your phone if needed. Why time is your best defense: Scams rely on speed. If you delay the scammer loses control, you regain clarity, and you reduce risk. There is rarely a legitimate situation that requires immediate payment within minutes. When in doubt ask a trusted adult, IT department, bank directly, or university administration. Asking is not weakness; it is smart behavior. Safe decision checklist: Am I reacting emotionally? Am I being pressured? Am I being asked for sensitive information? Have I verified independently? If unsure — delay action.`;

const R4_S1 = `Signs something may be wrong: You clicked a suspicious link, you shared OTP, your account logs you out suddenly, you receive password reset emails you did not request, unknown login alert, friends say you sent strange messages, money missing from account. Recognizing early signs reduces damage. First rule: Do not panic. Panic leads to poor decisions, more mistakes, responding to scammers, and sending more information. Instead: pause, breathe, act logically.`;

const R4_S2 = `Immediate protective steps if you clicked a suspicious link: 1) Disconnect from internet temporarily. 2) Close the website. 3) Do NOT enter further information. 4) Change affected passwords immediately. If you entered credentials change password immediately. Changing passwords safely: make it long (12+ characters), use new password (not reused), enable 2FA immediately. Change password for email (first priority), banking app, social media, and any reused accounts.`;

const R4_S3 = `If you shared OTP: Immediately contact your bank helpline, freeze account if needed, change password, enable 2FA. Time is critical in financial scams. If your social media is compromised (friends receive strange messages, posts appear that you did not write, password no longer works): use "Forgot Password", secure email first, enable 2FA, remove unknown devices, inform friends not to click links. If money has been sent: contact bank immediately, report transaction, freeze account if necessary, keep transaction record. Do not wait; the faster you act the better the chance of recovery.`;

const R4_S4 = `Reporting in Pakistan: You can report serious cyber incidents to FIA Cyber Crime Wing, your bank's official helpline, your university IT department, or platform support (Instagram, Facebook, WhatsApp). Reporting protects others too. Overcoming embarrassment: Many victims feel ashamed, stay silent, and do not report. Remember: scammers are professionals; anyone can be targeted. Reporting helps prevent future harm. When to ask for help: If unsure ask a trusted adult, IT department, bank, or university staff. It is better to ask early than regret later.`;

const R4_S5 = `Cleaning up after suspicion: After resolving issue — review privacy settings, check login history, remove unknown apps, update device software, enable 2FA everywhere. Think of it as digital cleaning. Learning from the experience: Instead of thinking "I was foolish" think "What did I learn?" Awareness improves with experience. Recovery flow summary: 1) Stay calm. 2) Disconnect if needed. 3) Change passwords. 4) Enable 2FA. 5) Contact bank if financial. 6) Report incident. 7) Inform affected contacts. Key habits: stay calm, change passwords, enable 2FA, contact official channels, report incidents.`;

const BASIC_2 = {
  courseTitle: "Recognizing Online Risks & Scams",
  description: "Awareness training on common online scams, suspicious message patterns, the Stop→Think→Verify→Act framework, and what to do if something feels wrong. Includes WhatsApp activity in the final module.",
  level: "basic",
  modules: [
    mod("Common Online Scams", [
      section("What is an Online Scam?", R1_S1),
      section("Fake Prize, Job & Bank Impersonation Scams", R1_S2),
      section("Bank, Loan & Investment Scams", R1_S3),
      section("Government, Delivery & Family Emergency Scams", R1_S4),
      section("Emotional Manipulation & Scam Checklist", R1_S5),
    ], [mcq("Should banks ever ask for OTP?", ["Yes", "No — never", "Only by phone"], 1), mcq("Why are prize scams suspicious?", ["Prizes are always real", "You never entered; they ask for a fee", "They use WhatsApp"], 1)]),
    mod("Suspicious Messages – What to Notice", [
      section("Types of Suspicious Messages", R2_S1),
      section("Warning Sign: Urgency & OTP/Password Requests", R2_S2),
      section("Suspicious Links & Unusual Requests", R2_S3),
      section("Unknown Authority & Too Good to Be True", R2_S4),
      section("Pause Rule, Verification & Checklist", R2_S5),
    ], [mcq("Why do scammers create urgency?", ["To be efficient", "To reduce thinking time and make you react emotionally", "To meet deadlines"], 1)]),
    mod("Safe Decision-Making Framework", [
      section("Why Decision-Making Matters & STEP 1 STOP", R3_S1),
      section("STEP 2 THINK", R3_S2),
      section("STEP 3 VERIFY", R3_S3),
      section("STEP 4 ACT & Applying the Framework", R3_S4),
      section("Emotional Control, Time & When in Doubt", R3_S5),
    ], [mcq("What is the first step in the framework?", ["Verify", "STOP", "Reply"], 1)]),
    mod("What To Do If Something Feels Wrong", [
      section("Signs Something May Be Wrong & Do Not Panic", R4_S1),
      section("Immediate Protective Steps & Changing Passwords", R4_S2),
      section("If You Shared OTP, Social Media Compromised or Money Sent", R4_S3),
      section("Reporting in Pakistan & Overcoming Embarrassment", R4_S4),
      section("Cleaning Up, Learning & Recovery Flow", R4_S5),
    ], [mcq("First rule if you suspect a scam?", ["Reply to sender", "Do not panic — stay calm and act logically", "Delete everything"], 1), mcq("Which account to secure first?", ["Social media", "Email", "Banking only"], 1)], "whatsapp"),
  ],
};

// ========== ADVANCED COURSE 1: Advanced Phishing Detection & Threat Analysis ==========
// 3 modules, up to 10 sections each, full content. Last module has email activity.
const A1_M1_S1 = `Phishing is one of the most common and dangerous cyber threats globally. Unlike basic scams, phishing attacks are carefully crafted, psychologically engineered, often technically disguised, and designed to bypass awareness. This module breaks down how phishing attacks are structured, how attackers think, how phishing messages are designed, and what stages exist in a phishing attack. Understanding the anatomy helps learners move from "I feel something is wrong" to "I can identify exactly what is happening."

Phishing is a social engineering attack in which an attacker impersonates a trusted entity to: steal credentials, steal financial data, install malware, gain unauthorized access, or escalate privileges. Unlike general scams, phishing is often targeted, strategically timed, and designed to mimic legitimate communication. The attacker "casts a bait" (fake message) hoping someone "bites" (clicks), submits credentials, or shares OTP. The more convincing the bait the higher the success rate.`;

const A1_M1_S2 = `The Phishing Attack Lifecycle – Most phishing attacks follow a predictable structure. Stage 1 Target Selection: Attacker chooses individuals (random mass phishing), employees of an organization, students, or banking customers. Advanced phishing often uses publicly available information to personalize messages. Stage 2 Impersonation: The attacker pretends to be bank, IT department, university admin, delivery service, CEO (in business email compromise), or government authority. Impersonation increases trust. Stage 3 Psychological Trigger: The message includes urgency ("Immediate action required"), fear ("Account suspended"), authority ("Security department"), reward ("You won"), or curiosity ("Confidential document shared"). This bypasses rational thinking. Stage 4 Action Request: The victim is asked to click link, download file, enter credentials, share OTP, or make payment. This is the attacker's goal stage. Stage 5 Exploitation: If the victim complies — credentials captured, account accessed, malware installed, money transferred, or internal systems breached.`;

const A1_M1_S3 = `Types of Phishing: Email Phishing – most common; fake login pages, fake invoices, fake account alerts. Smishing (SMS Phishing) – short messages, banking alerts, delivery scams, OTP theft attempts. Vishing (Voice Phishing) – fake bank calls, fake authority calls, threat-based manipulation. Spear Phishing – targeted, personalized, uses victim's name or role; often used against employees. Business Email Compromise (BEC) – fake CEO email, fake invoice, internal impersonation, financial transfer request.`;

const A1_M1_S4 = `Anatomy of a Phishing Email (structural breakdown): A phishing email typically contains 1) Display name (looks legitimate), 2) Suspicious sender address, 3) Urgent subject line, 4) Psychological pressure in body, 5) Action button/link, 6) Fake website clone. Example pattern: Subject "URGENT: Account Verification Required"; Opening "We detected suspicious activity…"; Pressure "Failure to act within 24 hours…"; Action "Click below to secure your account." This structure repeats across campaigns. Phishing attacks are 80% psychology. Common psychological levers: authority, urgency, scarcity, curiosity, fear, financial reward. Advanced learners must recognize emotional manipulation patterns.`;

const A1_M1_S5 = `Technical components (high-level): Phishing often includes fake login pages, URL spoofing, domain lookalikes, HTTPS misuse, logo replication, and email header manipulation. Advanced detection involves examining these elements. Why phishing works even on trained users: messages look realistic, timing aligns with real events, attackers research targets, cognitive overload reduces detection. Advanced defense requires pattern recognition, context awareness, verification habits, and technical inspection skills. Phishing vs normal scam: Basic scam often obvious and generic; phishing often subtle and can be personalized. Basic scam is emotion-heavy; phishing adds technical disguise. Phishing is more structured and layered. Defensive mindset shift: Beginner "Does this feel suspicious?" Advanced "What stage of phishing lifecycle is this?"`;

const A1_M2_S1 = `Understanding URLs: A URL is the web address of a site. A phishing URL may look similar but differ slightly. Anatomy of a URL: https://secure.bankname.com/login — https is protocol, secure is subdomain, bankname.com is main domain, /login is path. The main domain is the most important part. Critical rule: The domain is the part before .com (or .pk, .org). Attackers manipulate subdomains, hyphen placement, spelling, and extra words.`;

const A1_M2_S2 = `Common domain spoofing techniques: Lookalike domains – paypaI.com (capital I instead of L), micr0soft.com (zero instead of O), hbl-secure-login.com. Extra words added – bankname-support.com, bankname-verification.net, secure-bankname-alert.xyz. Legitimate organizations usually use officialname.com not long variations. Subdomain tricks: bankname.com.secure-login.ru — the real domain here is secure-login.ru; everything before it is misleading. Always read from right to left.`;

const A1_M2_S3 = `HTTPS is not enough: Many learners believe "If it has HTTPS it is safe." This is incorrect. HTTPS means the connection is encrypted. It does NOT mean the website is legitimate. Phishing sites can also use HTTPS. Sender address analysis: Phishing emails often manipulate sender fields. Example: Display name "National Bank of Pakistan", actual email nbp-support@gmail.com. Red flag: official banks do not use Gmail. Check: does domain match organization? Is it free email provider? Is spelling slightly off? Is there random string of numbers?`;

const A1_M2_S4 = `Basic email header awareness: Email headers contain sender information, routing path, and authentication checks. Email authentication methods include SPF, DKIM, DMARC. If these fail email may be spoofed. Suspicious attachments: Common malicious file types – .exe, .zip, .scr, .bat, .js. Office documents requesting "Enable Macros" from unknown sources are often malicious. Never enable macros from unknown sources. Link mismatch detection: Sometimes the visible text says www.bank.com but the actual link redirects to secure-bank-login.xyz. Always hover over links before clicking; on mobile long-press link to preview URL.`;

const A1_M2_S5 = `Technical red flags summary: Look for slight spelling differences, extra words in domain, suspicious TLD (.xyz, .ru, etc.), free email domains, mismatch between display name and email, urgent + link combination, unexpected attachments. Technical vs psychological indicators: Both matter – urgency (psychological) and domain mismatch (technical); fear and suspicious TLD; authority and sender spoofing; reward and attachment anomaly. Advanced detection requires both. Defensive mindset upgrade: Beginner "This feels suspicious." Advanced "I can explain why this domain is malicious." This prepares for phishing simulation labs and threat analysis.`;

const A1_M3_S1 = `Understanding phishing is theoretical; defending against phishing requires practice. This module introduces controlled phishing simulations, structured detection exercises, safe reporting workflows, and behavioral self-assessment. The goal is not to "trick" learners but to build detection confidence, strengthen analysis skills, improve reporting behavior, and reduce reaction time. A phishing simulation is a controlled training exercise where learners receive realistic but harmless phishing examples. Purpose: measure awareness, test detection skills, reinforce learning, reduce real-world risk. Simulations should be safe, not expose real data, not cause harm, and provide feedback. CyberShield simulation model: Exposure → Detection → Decision → Reporting → Feedback. The objective is improvement not punishment.`;

const A1_M3_S2 = `Simulation Type 1 – Email Phishing: Learners receive a simulated email e.g. Subject "URGENT: Account Suspension Notice", Body "We detected suspicious activity. Click below to verify." Learner must inspect sender email, hover link, identify domain anomaly, recognize urgency, decide not to click, and report email. Expected analytical output: identify psychological trigger (urgency), technical indicator (domain mismatch), impersonation attempt, and requested action. Simulation Type 2 – Smishing: Example SMS "Your parcel is held. Pay Rs. 2,000 here: secure-delivery-alert.xyz". Learner must recognize suspicious TLD, identify payment request, apply Stop→Think→Verify→Act, not click, and report. Simulation Type 3 – Vishing: Pre-recorded "Hello, this is your bank security team. Share OTP immediately." Learner must identify authority impersonation, recognize urgency, refuse OTP sharing, choose official verification.`;

const A1_M3_S3 = `Simulation difficulty levels: Level 1 – Obvious phishing (clear red flags). Level 2 – Moderately disguised (professional formatting + small domain variation). Level 3 – Advanced spear phishing (personalized details + subtle domain differences). This trains progressive detection. For each simulation learner must choose: Click link, Ignore, Report, or Verify externally. Correct action: Report + Verify independently. Performance metrics CyberShield LMS can track: click rate, time to decision, reporting behavior, confidence level, improvement over time.`;

const A1_M3_S4 = `Defensive reporting workflow: 1) Do not click. 2) Screenshot message. 3) Forward to security team. 4) Delete message. 5) Warn peers if necessary. Reporting is critical; many attacks escalate because they go unreported. Post-simulation reflection: What indicators did I miss? Which stage of lifecycle was this? Was psychological manipulation used? How could I respond faster? Reflection improves retention. Phishing awareness is not one-time training; periodic simulations and gradual difficulty increase build habit. Simulations must avoid humiliation, avoid public shaming, protect learner privacy, and focus on improvement. Psychological safety improves learning. Defensive mindset reinforcement: Which stage of lifecycle? Which technical indicator? What psychological trigger? What is the attacker's objective? This creates analytical defense capability.`;

const ADVANCED_1 = {
  courseTitle: "Advanced Phishing Detection & Threat Analysis",
  description: "Anatomy of phishing attacks, technical indicators (URLs, domains, headers), and phishing simulation and defensive practice. Includes email activity in the final module.",
  level: "advanced",
  modules: [
    mod("Anatomy of a Phishing Attack", [
      section("What is Phishing? (Advanced Definition)", A1_M1_S1),
      section("The Phishing Attack Lifecycle", A1_M1_S2),
      section("Types of Phishing", A1_M1_S3),
      section("Anatomy of a Phishing Email & Psychological Engineering", A1_M1_S4),
      section("Technical Components & Defensive Mindset", A1_M1_S5),
    ], [mcq("What are the five main stages of the phishing lifecycle?", ["Click, Pay, Share", "Target selection, Impersonation, Psychological trigger, Action request, Exploitation", "Send, Open, Reply"], 1)]),
    mod("Technical Indicators of Phishing", [
      section("Understanding URLs and Domains", A1_M2_S1),
      section("Domain Spoofing Techniques", A1_M2_S2),
      section("HTTPS and Sender Address Analysis", A1_M2_S3),
      section("Email Headers, Attachments & Link Mismatch", A1_M2_S4),
      section("Technical Red Flags & Defensive Mindset", A1_M2_S5),
    ], [mcq("What is the most important part of a URL?", ["Path", "Main domain", "Protocol"], 1), mcq("Does HTTPS guarantee safety?", ["Yes", "No — it only means encrypted connection", "Only for banks"], 1)]),
    mod("Phishing Simulation & Defensive Practice", [
      section("What is a Phishing Simulation & Philosophy", A1_M3_S1),
      section("Email, Smishing & Vishing Simulation Types", A1_M3_S2),
      section("Difficulty Levels, Correct Actions & Metrics", A1_M3_S3),
      section("Reporting Workflow, Reflection & Ethics", A1_M3_S4),
    ], [mcq("Purpose of phishing simulations?", ["To trick users", "To measure awareness and reinforce learning", "To test email only"], 1)], "email"),
  ],
};

// ========== ADVANCED COURSE 2 & 3 – full content, 5 sections per module (within 10 limit) ==========
const A2_M1_S1 = `When an email is sent: sender composes message, email client sends to mail server, mail server routes email, receiving mail server validates sender, message delivered or rejected. During this journey authentication checks occur. Without authentication spoofing becomes easy. Email spoofing is when an attacker sends an email pretending to be someone else, fakes the sender identity, or manipulates header information. Example: Display name "University IT Support", actual sender randomaddress@maliciousdomain.xyz. Spoofing exploits lack of authentication enforcement.`;

const A2_M1_S2 = `SPF (Sender Policy Framework) tells receiving servers which mail servers are allowed to send emails for this domain. If email comes from unauthorized server SPF fails. SPF protects against basic domain spoofing and unauthorized mail server use; however SPF alone is not enough. DKIM (DomainKeys Identified Mail) adds a digital signature to outgoing emails. This signature proves message was not altered and confirms sender domain authenticity. If content is modified signature fails. DKIM protects message integrity. DMARC (Domain-based Message Authentication, Reporting & Conformance) works on top of SPF and DKIM. It tells receiving servers what to do if authentication fails (nothing, quarantine, or reject) and generates reports to domain owners. Without these attackers can impersonate domains easily; with proper enforcement spoofed emails are blocked, marked as spam, or rejected.`;

const A2_M1_S3 = `Secure Email Gateways (SEGs): Organizations use SEGs to scan attachments, inspect links, filter spam, detect malicious patterns, and block suspicious senders. These systems use signature-based detection, heuristic analysis, machine learning, and reputation databases. They provide an additional defense layer. Link scanning and time-of-click protection: Advanced systems rewrite suspicious links, analyze URLs in sandbox environments, check website reputation, and monitor redirection chains. Some systems scan links again at click time to prevent delayed-activation phishing. Attachment sandboxing: When an attachment is received it may be opened in an isolated environment (sandbox); system monitors behavior and detects malicious activity. If suspicious behavior is detected the attachment is blocked.`;

const A2_M1_S4 = `Layered email defense model: Layer 1 – Authentication (SPF/DKIM/DMARC). Layer 2 – Spam filtering. Layer 3 – URL inspection. Layer 4 – Attachment scanning. Layer 5 – User awareness training. Layer 6 – Multi-factor authentication. No single layer is enough; defense in depth reduces risk. Even with strong email security some phishing emails pass filters, attackers constantly evolve, and human error still exists. User training and technical controls must work together. In universities, banks, and corporations email security policies may include mandatory 2FA, external sender warnings, restricted macro execution, domain monitoring, and phishing simulation programs. Security is continuous not one-time.`;

const A2_M1_S5 = `Interpreting email authentication results: When viewing email headers you may see SPF: Pass/Fail, DKIM: Pass/Fail, DMARC: Pass/Fail. If failures occur the email may be suspicious. Advanced learners should understand these elements; full header analysis is covered in the next module.`;

const A2_M2_S1 = `Email headers are technical details added during transmission. They include sending mail server, receiving mail server, timestamps, authentication results, Message ID, Reply-to address, and Return-path address. Headers cannot easily be faked entirely — inconsistencies appear. How to view headers: In Gmail click three dots → "Show Original". In Outlook File → Properties → Internet Headers. In most platforms "View Source" or "View Message Details". Key header fields to analyze: Return-Path, Received, From, Reply-To, SPF result, DKIM result, DMARC result, Message-ID.`;

const A2_M2_S2 = `Authentication results analysis: Look for SPF PASS/FAIL, DKIM PASS/FAIL, DMARC PASS/FAIL. Red flags: SPF FAIL, DKIM FAIL, DMARC FAIL, domain mismatch between From and Return-Path, suspicious mail server. Even one failed result requires deeper analysis. Tracing the sending server (Received fields): Emails include multiple "Received" entries showing which server handled the message and the IP address of the sending server. Read from bottom to top; the lowest "Received" entry shows origin server. Red flag example: Claimed sender bank.com but originating IP from unknown hosting provider in another country — mismatch increases suspicion.`;

const A2_M2_S3 = `From vs Reply-To mismatch: Example – From support@university.edu.pk but Reply-To security-alerts@gmail.com. This is a major red flag; legitimate institutions rarely use public email providers. Message-ID analysis: Message-ID contains domain reference. Example <123456@mail.randomdomain.xyz>. If domain differs from claimed sender investigate further. Structured threat investigation process (CyberShield Header Investigation Framework): 1) Confirm display sender. 2) Inspect actual sender domain. 3) Check authentication results. 4) Review routing path. 5) Analyze Reply-To. 6) Compare domains for alignment. 7) Document findings. This builds analyst discipline.`;

const A2_M2_S4 = `Lab Exercise 1 – Obvious spoof: Simulated header with SPF FAIL, DKIM FAIL, Reply-To mismatch, suspicious domain. Task: identify at least 3 technical red flags. Lab Exercise 2 – Moderate phishing: Simulation includes SPF PASS, DKIM PASS, DMARC PASS but suspicious lookalike domain. Learner must detect domain spoofing rather than authentication failure. Lab Exercise 3 – Spear phishing: Simulation includes correct domain, authentication passes, internal impersonation. Learner must detect contextual anomaly rather than technical failure; this prepares for Business Email Compromise cases.`;

const A2_M2_S5 = `Threat documentation template: Learners must document suspicious indicators, authentication results, domain mismatches, recommended action, and risk level assessment. Structured documentation builds professional skill. Limitations of header analysis: Some advanced attacks pass SPF/DKIM/DMARC; internal account compromise bypasses authentication; technical validation does not replace human judgment. Defense requires layered thinking. Escalation protocol: If header analysis confirms phishing — do not click, report to security team, quarantine message, warn affected users if necessary, block sending domain. In CyberShield LMS this may trigger incident response simulation.`;

const A2_M3_S1 = `Incident Response (IR) is a structured process used to detect security incidents, limit damage, investigate root cause, restore systems, and improve defenses. Phishing incidents are among the most common triggers for IR. Organizations typically follow six phases: Preparation, Detection, Containment, Eradication, Recovery, Lessons Learned. Phase 1 Preparation: Before an incident organizations deploy email security tools, enforce 2FA, train employees, conduct phishing simulations, establish reporting channels, and define incident response team. Preparation reduces response time.`;

const A2_M3_S2 = `Phase 2 Detection & Analysis: An incident may be detected through user reporting phishing email, security alert from email gateway, unusual login behavior, failed authentication attempts, or financial transaction anomaly. Security team analyzes email headers, affected users, authentication logs, and access history. Phase 3 Containment: Goal is to prevent further damage. Containment actions may include disabling compromised accounts, forcing password resets, blocking malicious domains, removing phishing email from inboxes, and revoking access tokens. Speed is critical.`;

const A2_M3_S3 = `Phase 4 Eradication: After containment remove the root cause — delete malicious attachments, remove malware from devices, patch vulnerabilities, update filters. This ensures the attacker cannot re-enter. Phase 5 Recovery: Restore user access, re-enable accounts securely, monitor for re-infection, reset credentials, reinforce MFA. Monitoring may continue for days or weeks. Phase 6 Lessons Learned: Analyze how did phishing bypass defenses? Which indicators were missed? Was reporting timely? Were users trained adequately? What improvements are required? This phase strengthens future defense.`;

const A2_M3_S4 = `Credential compromise scenario: Employee clicks phishing link and enters credentials. Response steps: immediately reset password, invalidate active sessions, enforce MFA, check login logs, investigate lateral movement, notify affected stakeholders. Financial fraud scenario: If phishing led to payment — contact financial institution, attempt transaction reversal, freeze account, report to authorities, review payment approval processes. Time directly affects recovery success. Organizational defense enhancements after incident: strengthen DMARC policy, tighten email filtering rules, add external sender banners, increase simulation frequency, conduct targeted retraining, implement stricter approval workflows.`;

const A2_M3_S5 = `User reporting culture: Encouraging early reporting reduces spread, protects other users, enables rapid containment, and improves detection speed. Healthy security culture avoids blame. Ethical and legal considerations: Organizations must protect user privacy, document incidents, comply with data protection laws, and report breaches if required. Incident response must follow legal framework. Advanced defensive mindset: How could this have been prevented? Which layer failed? Was authentication enforced? Was user awareness adequate? What structural change reduces risk? This mindset builds cybersecurity leadership capacity.`;

const ADVANCED_2 = {
  courseTitle: "Advanced Defensive Techniques & Email Security",
  description: "Email security architecture (SPF, DKIM, DMARC), email header analysis and threat investigation lab, and incident response and organizational phishing defense.",
  level: "advanced",
  modules: [
    mod("Email Security Architecture & Authentication Mechanisms", [
      section("How Email Travels & Email Spoofing", A2_M1_S1),
      section("SPF, DKIM & DMARC", A2_M1_S2),
      section("Secure Email Gateways, Link Scanning & Attachment Sandboxing", A2_M1_S3),
      section("Layered Defense & Organizational Context", A2_M1_S4),
      section("Interpreting Authentication Results", A2_M1_S5),
    ], [mcq("What problem does SPF solve?", ["Encryption", "Which mail servers can send for a domain", "Spam only"], 1)]),
    mod("Email Header Analysis & Threat Investigation Lab", [
      section("What Are Email Headers & How to View Them", A2_M2_S1),
      section("Authentication Results & Tracing Sending Server", A2_M2_S2),
      section("From vs Reply-To, Message-ID & Investigation Framework", A2_M2_S3),
      section("Lab Exercises 1, 2 & 3", A2_M2_S4),
      section("Documentation, Limitations & Escalation", A2_M2_S5),
    ], [mcq("Which authentication mechanisms should be checked?", ["Only SPF", "SPF, DKIM, DMARC", "Only DMARC"], 1)]),
    mod("Incident Response & Organizational Phishing Defense", [
      section("What is Incident Response & Phase 1 Preparation", A2_M3_S1),
      section("Phase 2 Detection & Phase 3 Containment", A2_M3_S2),
      section("Phase 4 Eradication, Phase 5 Recovery & Phase 6 Lessons Learned", A2_M3_S3),
      section("Credential & Financial Scenarios & Organizational Enhancements", A2_M3_S4),
      section("Reporting Culture, Ethics & Advanced Mindset", A2_M3_S5),
    ], [mcq("Six phases of incident response?", ["Detect, Delete, Done", "Preparation, Detection, Containment, Eradication, Recovery, Lessons Learned", "Report, Block, Forget"], 1)]),
  ],
};

// ========== ADVANCED COURSE 3: Threat Intelligence & Advanced Social Engineering Defense ==========
const A3_M1_S1 = `Threat intelligence is actionable information about threats, attackers, and risks that helps organizations make better security decisions. It answers: Who is targeting us? What tactics are being used? Which vulnerabilities are being exploited? How can we prepare in advance? Threat intelligence shifts defense from reactive to proactive. Types: Strategic intelligence – high-level trends, industry-specific and regional threats, geopolitical cyber risks (e.g. increase in banking phishing in a country); used by executives and policymakers. Tactical intelligence – phishing templates, malware distribution methods, attack patterns, social engineering trends; used by security teams. Operational intelligence – real-time active domains, malicious IPs, ongoing campaigns, IOCs; used by IR teams.`;

const A3_M1_S2 = `The adversary mindset (defensive understanding): To defend effectively you must understand how attackers think. Attackers ask: Who is vulnerable? What information is publicly available? What emotional triggers work? Which timing increases success? Which department has financial authority? Understanding these questions strengthens defense. Reconnaissance in social engineering: Before launching a targeted phishing campaign attackers may gather employee names, job titles, organizational hierarchy, public email formats, LinkedIn data, public announcements, and recent company events. This phase is called reconnaissance. Recognizing what information is publicly exposed helps reduce risk.`;

const A3_M1_S3 = `Social engineering campaign structure: Information gathering → Target profiling → Message crafting → Timing optimization → Delivery → Exploitation. This is similar to phishing lifecycle but more targeted. Defenders must disrupt these stages. Why public information matters: Oversharing on LinkedIn, Instagram, Facebook, or company websites can provide attackers organizational structure, internal terminology, project names, and executive relationships. Reducing public exposure reduces attacker advantage. Psychological profiling in advanced attacks: Advanced attackers may target finance teams (invoice fraud), HR (resume malware), executives (BEC), or IT (credential harvesting). They tailor messages based on role — targeted social engineering.`;

const A3_M1_S4 = `Defensive counter-intelligence mindset: What information about us is publicly available? What common emotional triggers affect our users? Which departments are most targeted? Do we simulate role-based phishing? Do we monitor new suspicious domains? Threat intelligence enables structured prevention. Indicators of compromise (IOCs): Malicious domains, suspicious IP addresses, hash values of malware, email subject patterns, sender patterns. Security teams use IOCs to block known threats. Continuous threat monitoring: Threat landscape evolves daily. Organizations must subscribe to threat feeds, monitor domain registrations, track phishing campaigns, update filtering rules, and share intelligence internally. Cybersecurity is dynamic.`;

const A3_M1_S5 = `From reactive to proactive defense: Beginner "I clicked a phishing email." Intermediate "I detected phishing." Advanced "I anticipated the phishing campaign and blocked it." Threat intelligence enables anticipation. Real-world example: Suppose threat intelligence reports increase in fake university admission emails targeting students. Defensive actions may include sending awareness email to students, increasing email filtering rules, blocking similar domains, running targeted simulation. Intelligence informs defense strategy. Building a threat-aware culture requires continuous monitoring, cross-team communication, executive awareness, user training, regular simulation, and adaptive policies. Threat intelligence is not just technical; it is organizational.`;

const A3_M2_S1 = `Advanced social engineering is targeted (not random), personalized, context-aware, often multi-stage, and designed to bypass security filters. It may involve internal impersonation, prior reconnaissance, real-world timing alignment, and multi-channel communication. Business Email Compromise (BEC) is a highly targeted form of phishing. It often involves impersonating executives or finance department, requesting urgent payment, or changing vendor bank details. Example: Email appears from "CEO" — "Transfer funds urgently for confidential acquisition." Characteristics: urgent tone, authority pressure, confidentiality request, bypasses normal approval process. Red flags in BEC: unusual payment request, breaking standard procedures, pressure for secrecy, slight email domain variation, executive tone mismatch.`;

const A3_M2_S2 = `Pretexting is when an attacker creates a believable story (pretext) to extract information. Examples: pretending to be IT support, HR, vendor, or law enforcement. The attacker builds credibility first then asks for sensitive information. Defensive awareness: Always verify identity through official channels. Never rely solely on caller ID, email display name, or social media identity. Multi-stage social engineering: Stage 1 innocent conversation, Stage 2 trust building, Stage 3 information extraction, Stage 4 financial or credential exploitation. Example: LinkedIn connection → casual discussion → sending "project document" → malicious attachment. This gradual escalation increases success rate.`;

const A3_M2_S3 = `Multi-channel attacks: Attackers may combine email, SMS, phone calls, social media, and messaging apps. Example: Email sent; follow-up call "Did you receive my email?" This increases credibility. Authority and organizational hierarchy exploitation: Attackers exploit fear of authority, respect for executives, and junior employee hesitation. Example "Do not discuss this with anyone" — isolation increases manipulation success. Psychological levers in advanced attacks: authority, scarcity, confidentiality, reciprocity, familiarity, social proof. Recognizing these levers reduces susceptibility.`;

const A3_M2_S4 = `Defensive countermeasures (individual): Verify financial requests independently, follow standard procedures, refuse secrecy pressure, use call-back verification, confirm via known contact method, report suspicious requests immediately. Defensive countermeasures (organizational): Dual-approval for payments, vendor bank change verification, mandatory MFA, external sender banners, strict privilege separation, executive impersonation detection tools. Structure reduces manipulation success. Zero-trust communication model: "Never trust solely based on identity claim." Always verify through independent channel — even if message appears internal.`;

const A3_M2_S5 = `Role-based risk awareness: Finance → invoice fraud; HR → resume malware; IT → credential harvesting; Executives → spear phishing; Students → scholarship scams. Training should be contextual. Defensive scenario exercise: Finance officer receives email from "CFO" requesting urgent vendor payment. Tasks: identify social engineering tactics, identify red flags, propose verification method, suggest organizational control. This builds applied defense thinking. Building resistance requires structured verification habits, clear escalation pathways, psychological awareness, organizational procedure compliance, and culture that supports questioning authority safely. Security culture reduces manipulation success.`;

const A3_M3_S1 = `This capstone challenges learners to apply everything learned. Instead of isolated theory learners will analyze a simulated real-world phishing campaign, investigate technical indicators, examine authentication results, identify social engineering tactics, apply incident response strategy, and propose preventive controls. This module validates advanced defensive capability. Capstone structure: 1) Simulated phishing email, 2) Email header dataset, 3) User action timeline, 4) Organizational context, 5) Threat intelligence update, 6) Investigation report submission.`;

const A3_M3_S2 = `Scenario overview: A university receives a phishing email targeting finance department employees. Subject "Urgent Vendor Payment Update – Confidential." Three employees receive the email; one clicks the link, one reports it, one ignores it. You are part of the security investigation team. Phase 1 Initial email analysis: Identify impersonation attempt, psychological triggers, requested action, suspicious language, domain inconsistencies. Phase 2 Header investigation: Given simulated header output with SPF/DKIM/DMARC PASS but suspicious lookalike domain and Reply-To mismatch. Tasks: identify domain alignment issue, analyze routing path, identify spoofing attempt, document findings.`;

const A3_M3_S3 = `Phase 3 Threat intelligence correlation: Threat feed reports increase in vendor invoice phishing targeting universities. Learners must connect incident to broader campaign, assess whether this is isolated or campaign-based, and identify IOCs. Phase 4 Incident response decision: One employee entered credentials. Learners must propose immediate containment steps, credential reset procedure, access log review, session invalidation, and MFA enforcement. Phase 5 Organizational remediation: Recommend email filter updates, domain blocking, additional training, policy reinforcement, and payment approval changes. This tests strategic thinking.`;

const A3_M3_S4 = `Risk assessment exercise: Assign threat severity (Low/Medium/High/Critical), business impact level, data exposure risk, financial risk level. Justification required. Investigation report template: Submit structured report including 1) Executive Summary, 2) Attack Description, 3) Technical Findings, 4) Psychological Analysis, 5) Authentication Review, 6) Containment Actions, 7) Preventive Recommendations, 8) Risk Rating. This builds professional documentation skills. Advanced analytical reflection: Which defensive layer failed? Which stage of phishing lifecycle was most critical? How could threat intelligence have prevented this? What policy improvements are required?`;

const A3_M3_S5 = `Defensive maturity evaluation: Learners are evaluated on technical accuracy, structured reasoning, correct incident response, clear documentation, and strategic prevention planning. Capstone completion criteria: To pass Advanced Path — 70%+ on technical identification, correct containment strategy, proper authentication interpretation, complete structured report. Certification: CyberShield Advanced Phishing Defense Specialist.`;

const ADVANCED_3 = {
  courseTitle: "Threat Intelligence & Advanced Social Engineering Defense",
  description: "Foundations of threat intelligence and adversary mindset, advanced social engineering tactics and defensive countermeasures, and capstone threat intelligence and phishing investigation challenge.",
  level: "advanced",
  modules: [
    mod("Foundations of Threat Intelligence & Adversary Mindset", [
      section("What is Threat Intelligence & Types", A3_M1_S1),
      section("Adversary Mindset & Reconnaissance", A3_M1_S2),
      section("Campaign Structure & Psychological Profiling", A3_M1_S3),
      section("Defensive Counter-Intelligence & IOCs", A3_M1_S4),
      section("Proactive Defense & Threat-Aware Culture", A3_M1_S5),
    ], [mcq("What is threat intelligence?", ["Spy software", "Actionable information about threats for security decisions", "Email filtering only"], 1)]),
    mod("Advanced Social Engineering Tactics & Defensive Countermeasures", [
      section("What Makes Social Engineering Advanced & BEC", A3_M2_S1),
      section("Pretexting & Multi-Stage Attacks", A3_M2_S2),
      section("Multi-Channel Attacks & Psychological Levers", A3_M2_S3),
      section("Individual & Organizational Countermeasures & Zero-Trust", A3_M2_S4),
      section("Role-Based Risk & Building Resistance", A3_M2_S5),
    ], [mcq("What is zero-trust communication?", ["No email", "Never trust identity claim alone; verify independently", "Encryption only"], 1)]),
    mod("Capstone – Threat Intelligence & Phishing Investigation Challenge", [
      section("Capstone Overview & Structure", A3_M3_S1),
      section("Scenario, Phase 1 & Phase 2", A3_M3_S2),
      section("Phase 3, 4 & 5", A3_M3_S3),
      section("Risk Assessment & Investigation Report", A3_M3_S4),
      section("Defensive Maturity & Completion Criteria", A3_M3_S5),
    ], [mcq("Most critical step after credential compromise?", ["Wait", "Immediate containment and credential reset", "Email attacker"], 1)]),
  ],
};

const ALL_COURSES = [BASIC_1, BASIC_2, ADVANCED_1, ADVANCED_2, ADVANCED_3];

async function run() {
  try {
    await mongoose.connect(process.env.MONGODB_URI || "mongodb://localhost:27017/cybershield");
    console.log("Connected to MongoDB");
    const user = await User.findOne().select("_id displayName email").lean();
    if (!user) {
      console.error("No user in DB. Create at least one user, then run the seed again.");
      process.exit(1);
    }
    const force = process.argv.includes("--force");
    for (const c of ALL_COURSES) {
      const existing = await Course.findOne({ courseTitle: c.courseTitle, orgId: null });
      if (existing && !force) {
        console.log("Skip (exists):", c.courseTitle);
        continue;
      }
      if (existing && force) await Course.deleteOne({ _id: existing._id });
      await Course.create({
        courseTitle: c.courseTitle,
        description: c.description,
        level: c.level,
        modules: c.modules,
        createdBy: user._id,
        createdByName: user.displayName || "Seed",
        createdByEmail: user.email || "seed@local",
        badges: [],
        orgId: null,
      });
      console.log("Seeded:", c.courseTitle);
    }
    console.log("Seed done.");
    process.exit(0);
  } catch (err) {
    console.error("Seed error:", err);
    process.exit(1);
  }
}

run();
