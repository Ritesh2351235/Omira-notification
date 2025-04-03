const express = require('express');
const dotenv = require('dotenv');
const { OpenAI } = require('openai');
const winston = require('winston');
const path = require('path');

dotenv.config();

// Initialize Express app
const app = express();
app.use(express.json());

// Constants
const TRIGGER_PHRASES = ["hey omi", "hey, omi"]; // Base triggers
const PARTIAL_FIRST = ["hey", "hey,"]; // First part of trigger
const PARTIAL_SECOND = ["omi"]; // Second part of trigger
const QUESTION_AGGREGATION_TIME = 5000; // 5 seconds to wait for collecting the question
const NOTIFICATION_COOLDOWN = 60000; // 1 minute cooldown between notifications
// Add OpenAI API rate limiting constants
const MAX_CONCURRENT_REQUESTS = 10; // Maximum concurrent OpenAI requests
const RETRY_DELAY = 2000; // Initial retry delay in ms
const MAX_RETRIES = 3; // Maximum number of retries for OpenAI calls

// Create logs directory if it doesn't exist
const logDir = path.join(__dirname, 'logs');
require('fs').mkdirSync(logDir, { recursive: true });

// Configure Winston logger
const logger = winston.createLogger({
  level: 'debug',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.printf(({ timestamp, level, message, ...meta }) => {
      return `${timestamp} - ${level.toUpperCase()} - [${meta.threadName || 'main'}] - ${meta.module || 'app'}:${meta.lineno || '?'} - ${message}`;
    })
  ),
  transports: [
    new winston.transports.File({ filename: path.join(logDir, 'mentor.log') }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

if (!process.env.OPENAI_API_KEY) {
  logger.error("OPENAI_API_KEY not found in environment variables");
  throw new Error("OPENAI_API_KEY environment variable is required");
}

// Simple request queue for OpenAI calls
const openaiRequestQueue = {
  activeRequests: 0,
  queue: [],

  async add(fn) {
    if (this.activeRequests < MAX_CONCURRENT_REQUESTS) {
      return this.executeRequest(fn);
    } else {
      return new Promise((resolve, reject) => {
        this.queue.push({ fn, resolve, reject });
      });
    }
  },

  async executeRequest(fn) {
    this.activeRequests++;
    try {
      return await fn();
    } finally {
      this.activeRequests--;
      this.processNextRequest();
    }
  },

  processNextRequest() {
    if (this.queue.length > 0 && this.activeRequests < MAX_CONCURRENT_REQUESTS) {
      const { fn, resolve, reject } = this.queue.shift();
      this.executeRequest(fn).then(resolve).catch(reject);
    }
  }
};

class MessageBuffer {
  constructor() {
    logger.info("Initializing MessageBuffer");
    this.buffers = new Map();
    this.cleanupInterval = 300000; // 5 minutes for cleanup check
    this.lastCleanup = Date.now();
    this.silenceThreshold = 120000; // 2 minutes silence threshold
    this.notificationCooldowns = new Map(); // Track cooldowns for notifications
    logger.debug(`MessageBuffer initialized with cleanup_interval=${this.cleanupInterval}, silence_threshold=${this.silenceThreshold}`);
  }

  getBuffer(sessionId) {
    logger.debug(`Getting buffer for session_id: ${sessionId}`);
    const currentTime = Date.now();

    // Cleanup old sessions periodically
    if (currentTime - this.lastCleanup > this.cleanupInterval) {
      logger.info("Triggering cleanup of old sessions");
      this.cleanupOldSessions();
    }

    if (!this.buffers.has(sessionId)) {
      logger.info(`Creating new buffer for session_id: ${sessionId}`);
      this.buffers.set(sessionId, {
        messages: [], // For context tracking
        lastActivity: currentTime,
        // Add trigger related fields
        triggerDetected: false,
        triggerTime: 0,
        collectedQuestion: [],
        responseSent: false,
        partialTrigger: false,
        partialTriggerTime: 0,
        lastNotification: 0 // Track when the last notification was sent
      });
    } else {
      const buffer = this.buffers.get(sessionId);
      const timeSinceActivity = currentTime - buffer.lastActivity;

      // Only reset context if silence threshold is reached
      if (timeSinceActivity > this.silenceThreshold) {
        logger.info(`Silence period detected for session ${sessionId}. Time since activity: ${timeSinceActivity / 1000}s`);
        // Reset trigger states on silence, but keep messages for history
        buffer.triggerDetected = false;
        buffer.triggerTime = 0;
        buffer.collectedQuestion = [];
        buffer.responseSent = false;
        buffer.partialTrigger = false;
      }

      buffer.lastActivity = currentTime;
    }

    return this.buffers.get(sessionId);
  }

  cleanupOldSessions() {
    logger.info("Starting cleanup of old sessions");
    const currentTime = Date.now();
    const initialCount = this.buffers.size;

    for (const [sessionId, data] of this.buffers.entries()) {
      if (currentTime - data.lastActivity > 3600000) { // 1 hour
        logger.info(`Removing expired session: ${sessionId}`);
        this.buffers.delete(sessionId);
        this.notificationCooldowns.delete(sessionId);
      }
    }

    this.lastCleanup = currentTime;
    logger.info(`Cleanup complete. Removed ${initialCount - this.buffers.size} sessions. Active sessions: ${this.buffers.size}`);
  }

  isCooldownActive(sessionId) {
    const currentTime = Date.now();
    const buffer = this.buffers.get(sessionId);
    if (!buffer) return false;

    const lastNotification = buffer.lastNotification || 0;
    const timeSince = currentTime - lastNotification;

    if (timeSince < NOTIFICATION_COOLDOWN) {
      logger.info(`Cooldown active for session ${sessionId}. ${(NOTIFICATION_COOLDOWN - timeSince) / 1000}s remaining`);
      return true;
    }
    return false;
  }

  setCooldown(sessionId) {
    const buffer = this.buffers.get(sessionId);
    if (buffer) {
      buffer.lastNotification = Date.now();
    }
  }
}

// Initialize message buffer
const messageBuffer = new MessageBuffer();

// Updated OpenAI response function with retries and queue
async function getOpenAIResponse(text, messages) {
  return openaiRequestQueue.add(async () => {
    logger.info(`Sending question to OpenAI: ${text} (Active requests: ${openaiRequestQueue.activeRequests}, Queue size: ${openaiRequestQueue.queue.length})`);

    // Format message history for context
    const messageHistory = messages.slice(-10).map(msg => {
      return {
        role: msg.is_user ? "user" : "assistant",
        content: msg.text
      };
    });

    const systemPrompt = `You are Omira, an AI mentor that provides precise, helpful advice. 
    You analyze conversations and provide personalized insights.
    Keep responses under 300 characters, be direct, and use simple language.
    Focus on actionable advice based on the conversation context.
    End with a thoughtful question that helps the user implement your advice.`;

    // Implement retries with exponential backoff
    let lastError = null;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        const response = await openai.chat.completions.create({
          model: "gpt-4",
          messages: [
            {
              role: "system",
              content: systemPrompt
            },
            ...messageHistory,
            {
              role: "user",
              content: text
            }
          ],
          temperature: 0.7,
          max_tokens: 150,
        });

        const answer = response.choices[0].message.content.trim();
        logger.info(`Received response from OpenAI: ${answer}`);
        return answer;
      } catch (error) {
        lastError = error;

        // Handle rate limiting specially
        if (error.status === 429 || error.message?.includes('rate limit')) {
          const delay = RETRY_DELAY * Math.pow(2, attempt - 1);
          logger.warn(`OpenAI rate limit hit. Retrying in ${delay}ms (attempt ${attempt}/${MAX_RETRIES})`);

          // Wait before next attempt
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }

        // For other errors, retry only on first attempt
        if (attempt < 2 && (error.status >= 500 || error.status === 0)) {
          logger.warn(`OpenAI request failed with status ${error.status}. Retrying once.`);
          await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
          continue;
        }

        // For other errors or final attempt, throw
        throw error;
      }
    }

    // If we get here, we've exhausted all retries
    logger.error(`OpenAI request failed after ${MAX_RETRIES} attempts: ${lastError.message}`, { error: lastError });
    return "I'm sorry, I encountered an error processing your request. Please try again later.";
  });
}

// Update analyzeConversation to use the queue and retry logic
async function analyzeConversation(messages) {
  return openaiRequestQueue.add(async () => {
    logger.info(`Analyzing conversation with ${messages.length} messages (Active requests: ${openaiRequestQueue.activeRequests}, Queue size: ${openaiRequestQueue.queue.length})`);

    if (messages.length < 3) {
      logger.info("Not enough messages to analyze");
      return null;
    }

    // Format the conversation for analysis
    const conversationText = messages.map(msg => {
      const speaker = msg.is_user ? "User" : "Other Person";
      return `${speaker}: ${msg.text}`;
    }).join("\n");

    const systemPrompt = `You are an AI mentor that provides precise, helpful advice based on conversations.
    
    Evaluate if this conversation warrants sending a notification by checking ALL these conditions:
    1. The conversation contains specific problems, challenges, goals, or questions
    2. You have a STRONG, CLEAR opinion that would significantly help the situation
    3. The insight is important enough to interrupt for
    4. Your advice would be truly valuable right now
    
    If ANY condition is NOT met, return { "shouldNotify": false, "reason": "reason here" }.
    
    If ALL conditions are met, return:
    {
      "shouldNotify": true,
      "notification": "Your notification message here"
    }
    
    Your notification must:
    - Speak directly to the user
    - Take a clear stance - no hedging language
    - Keep it under 300 characters
    - Use simple, everyday words
    - Reference specific details from the conversation
    - Be bold and direct
    - End with a specific question about implementing your advice
    
    Conversation:
    ${conversationText}`;

    // Implement retries with exponential backoff
    let lastError = null;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        const response = await openai.chat.completions.create({
          model: "gpt-4",
          messages: [
            {
              role: "system",
              content: systemPrompt
            }
          ],
          temperature: 0.3,
          response_format: { type: "json_object" },
          max_tokens: 400
        });

        try {
          const result = JSON.parse(response.choices[0].message.content.trim());
          logger.info(`Analysis result: shouldNotify=${result.shouldNotify}`);
          if (!result.shouldNotify) {
            logger.info(`Reason for not notifying: ${result.reason}`);
          }
          return result;
        } catch (e) {
          logger.error(`Error parsing analysis result: ${e.message}`);
          return { shouldNotify: false, reason: "Error parsing result" };
        }
      } catch (error) {
        lastError = error;

        // Handle rate limiting specially
        if (error.status === 429 || error.message?.includes('rate limit')) {
          const delay = RETRY_DELAY * Math.pow(2, attempt - 1);
          logger.warn(`OpenAI rate limit hit during analysis. Retrying in ${delay}ms (attempt ${attempt}/${MAX_RETRIES})`);

          // Wait before next attempt
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }

        // For other errors, retry only on first attempt
        if (attempt < 2 && (error.status >= 500 || error.status === 0)) {
          logger.warn(`OpenAI request failed with status ${error.status}. Retrying once.`);
          await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
          continue;
        }

        // For other errors or final attempt, return error result
        logger.error(`Error analyzing conversation: ${error.message}`, { error });
        return { shouldNotify: false, reason: "Error in analysis" };
      }
    }

    // If we get here, we've exhausted all retries
    logger.error(`OpenAI analysis request failed after ${MAX_RETRIES} attempts: ${lastError.message}`, { error: lastError });
    return { shouldNotify: false, reason: "Error in analysis after multiple attempts" };
  });
}

// Webhook endpoints
app.post('/webhook', async (req, res) => {
  logger.info("Received webhook POST request");
  try {
    const data = req.body;
    const sessionId = data.session_id;
    const segments = data.segments || [];
    let messageId = data.message_id;
    let hasProcessed = false;

    if (!messageId) {
      messageId = `${sessionId}_${Date.now()}`;
      logger.info(`Generated message_id: ${messageId}`);
    }

    logger.info(`Processing webhook for session_id: ${sessionId}, message_id: ${messageId}, segments count: ${segments.length}`);

    if (!sessionId) {
      logger.error("No session_id provided in request");
      return res.status(400).json({ message: "No session_id provided" });
    }

    const currentTime = Date.now();
    const buffer = messageBuffer.getBuffer(sessionId);

    // Process new messages
    logger.info(`Processing ${segments.length} segments for session ${sessionId}`);
    for (const segment of segments) {
      if (!segment.text || hasProcessed) {
        logger.debug("Skipping empty segment or already processed");
        continue;
      }

      const text = segment.text.trim();
      const textLower = text.toLowerCase();

      if (text) {
        const timestamp = segment.start || currentTime;
        const isUser = segment.is_user || false;
        logger.info(`Processing segment - is_user: ${isUser}, timestamp: ${timestamp}, text: ${text.substring(0, 50)}...`);

        // Add message to buffer (for context tracking)
        buffer.messages.push({
          text: text, // Use original case
          timestamp,
          is_user: isUser
        });

        // Check for complete trigger phrases
        if (!buffer.triggerDetected && TRIGGER_PHRASES.some(trigger => textLower.includes(trigger))) {
          logger.info(`Complete trigger phrase detected in session ${sessionId}`);
          buffer.triggerDetected = true;
          buffer.triggerTime = currentTime;
          buffer.collectedQuestion = [];
          buffer.responseSent = false;
          buffer.partialTrigger = false;

          // Extract any question part that comes after the trigger
          const questionPart = textLower.includes('omi,') ?
            text.split(/omi,/i)[1]?.trim() :
            textLower.includes('omi') ?
              text.split(/omi/i)[1]?.trim() : '';

          // CHANGE: Immediately process if we have a question with the trigger phrase
          if (questionPart) {
            buffer.collectedQuestion.push(questionPart);
            logger.info(`Collected question part from trigger: ${questionPart}`);

            // Immediately process the question without waiting
            logger.info(`Processing immediate question from trigger: ${questionPart}`);
            const response = await getOpenAIResponse(questionPart, buffer.messages.slice(-10));
            logger.info(`Got immediate response from OpenAI: ${response}`);

            // Reset trigger states
            buffer.triggerDetected = false;
            buffer.triggerTime = 0;
            buffer.collectedQuestion = [];
            buffer.responseSent = true;
            buffer.partialTrigger = false;
            messageBuffer.setCooldown(sessionId);
            hasProcessed = true;

            return res.status(200).json({ message: response });
          }

          // If no question part, analyze conversation
          if (!questionPart && !messageBuffer.isCooldownActive(sessionId)) {
            logger.info(`Trigger with no question detected, analyzing conversation`);
            const analysis = await analyzeConversation(buffer.messages.slice(-20));

            if (analysis && analysis.shouldNotify) {
              // Send the notification
              messageBuffer.setCooldown(sessionId);
              logger.info(`Sending notification: ${analysis.notification}`);
              return res.status(200).json({ message: analysis.notification });
            } else {
              logger.info(`No notification needed: ${analysis?.reason || 'No analysis result'}`);
              return res.status(200).json({ status: "success", message: "I'm listening, but I don't have any insights to share right now." });
            }
          }

          continue;
        }

        // Check for partial triggers
        if (!buffer.triggerDetected) {
          // Check for first part of trigger
          if (PARTIAL_FIRST.some(part => textLower.endsWith(part))) {
            logger.info(`First part of trigger detected in session ${sessionId}`);
            buffer.partialTrigger = true;
            buffer.partialTriggerTime = currentTime;
            continue;
          }

          // Check for second part if we're waiting for it
          if (buffer.partialTrigger) {
            const timeSincePartial = currentTime - buffer.partialTriggerTime;
            if (timeSincePartial <= 2000) { // 2 second window to complete the trigger
              if (PARTIAL_SECOND.some(part => textLower.includes(part))) {
                logger.info(`Complete trigger detected across segments in session ${sessionId}`);
                buffer.triggerDetected = true;
                buffer.triggerTime = currentTime;
                buffer.collectedQuestion = [];
                buffer.responseSent = false;
                buffer.partialTrigger = false;

                // Extract any question part that comes after "omi"
                const questionPart = textLower.includes('omi,') ?
                  text.split(/omi,/i)[1]?.trim() :
                  textLower.includes('omi') ?
                    text.split(/omi/i)[1]?.trim() : '';

                // CHANGE: Immediately process if we have a question with the trigger phrase
                if (questionPart) {
                  buffer.collectedQuestion.push(questionPart);
                  logger.info(`Collected question part from second trigger part: ${questionPart}`);

                  // Immediately process the question without waiting
                  logger.info(`Processing immediate question from partial trigger: ${questionPart}`);
                  const response = await getOpenAIResponse(questionPart, buffer.messages.slice(-10));
                  logger.info(`Got immediate response from OpenAI: ${response}`);

                  // Reset trigger states
                  buffer.triggerDetected = false;
                  buffer.triggerTime = 0;
                  buffer.collectedQuestion = [];
                  buffer.responseSent = true;
                  buffer.partialTrigger = false;
                  messageBuffer.setCooldown(sessionId);
                  hasProcessed = true;

                  return res.status(200).json({ message: response });
                } else if (!messageBuffer.isCooldownActive(sessionId)) {
                  // Trigger with no question - analyze conversation
                  logger.info(`Trigger with no question detected, analyzing conversation`);
                  const analysis = await analyzeConversation(buffer.messages.slice(-20));

                  if (analysis && analysis.shouldNotify) {
                    // Send the notification
                    messageBuffer.setCooldown(sessionId);
                    logger.info(`Sending notification: ${analysis.notification}`);
                    return res.status(200).json({ message: analysis.notification });
                  } else {
                    logger.info(`No notification needed: ${analysis?.reason || 'No analysis result'}`);
                    return res.status(200).json({ status: "success", message: "I'm listening, but I don't have any insights to share right now." });
                  }
                }
                continue;
              }
            } else {
              // Reset partial trigger if too much time has passed
              buffer.partialTrigger = false;
            }
          }
        }

        // If trigger was detected, collect the question
        if (buffer.triggerDetected && !buffer.responseSent && !hasProcessed) {
          const timeSinceTrigger = currentTime - buffer.triggerTime;
          logger.info(`Time since trigger: ${timeSinceTrigger / 1000} seconds`);

          if (timeSinceTrigger <= QUESTION_AGGREGATION_TIME) {
            buffer.collectedQuestion.push(text);
            logger.info(`Collecting question part: ${text}`);
            logger.info(`Current collected question: ${buffer.collectedQuestion.join(' ')}`);
          }

          // Check if we should process the question
          const shouldProcess = (
            (timeSinceTrigger > QUESTION_AGGREGATION_TIME && buffer.collectedQuestion.length > 0) ||
            (buffer.collectedQuestion.length > 0 && textLower.includes('?')) ||
            (timeSinceTrigger > QUESTION_AGGREGATION_TIME * 1.5)
          );

          if (shouldProcess && buffer.collectedQuestion.length > 0) {
            // Process question and send response
            let fullQuestion = buffer.collectedQuestion.join(' ').trim();

            logger.info(`Processing complete question: ${fullQuestion}`);
            const response = await getOpenAIResponse(fullQuestion, buffer.messages.slice(-10));
            logger.info(`Got response from OpenAI: ${response}`);

            // Reset trigger states
            buffer.triggerDetected = false;
            buffer.triggerTime = 0;
            buffer.collectedQuestion = [];
            buffer.responseSent = true;
            buffer.partialTrigger = false;
            messageBuffer.setCooldown(sessionId);
            hasProcessed = true;

            return res.status(200).json({ message: response });
          }
        }
      }
    }

    // Return success if no response needed
    logger.debug("No response needed at this time");
    return res.status(200).json({ status: "success" });

  } catch (error) {
    logger.error(`Error processing webhook: ${error.message}`, { error });
    return res.status(500).json({ error: "Internal server error" });
  }
});

app.get('/webhook/setup-status', (req, res) => {
  logger.debug("Received setup-status GET request");
  return res.status(200).json({ is_setup_completed: true });
});

app.get('/status', (req, res) => {
  logger.debug("Received status GET request");
  const activeSessions = messageBuffer.buffers.size;
  const uptime = Date.now() - startTime;
  logger.info(`Status check - Active sessions: ${activeSessions}, Uptime: ${uptime / 1000}s`);
  return res.status(200).json({
    active_sessions: activeSessions,
    uptime: uptime
  });
});

// Track start time
const startTime = Date.now();
logger.info(`Application initialized. Start time: ${new Date(startTime).toISOString()}`);

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, '0.0.0.0', () => {
  logger.info(`Server running on port ${PORT}`);
});

// Export for testing
module.exports = { app, messageBuffer };
