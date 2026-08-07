// Needed for dotenv
require("dotenv").config();

// Needed for Express
var express = require('express');
var app = express();
var multer = require('multer');
var upload = multer({ dest: 'uploads/' });

// Needed for form-data and fetch
const FormData = require('form-data');
const fetch = require('node-fetch');
const fs = require('fs');

// Needed for EJS
app.set('view engine', 'ejs');

// Needed for public directory
app.use(express.static(__dirname + '/public'));

// Needed for parsing form data
app.use(express.json());
app.use(express.urlencoded({extended: true}));

// Turns the LLM's lightweight markdown (**bold**, -/1. lists, line breaks)
// into safe HTML for display. Escapes first so no raw HTML from the model
// or source document can slip through.
function escapeHtml(str) {
    return str.replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
}

function applyInlineMarkdown(str) {
    return str.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

function formatResult(text) {
    if (!text) return null;

    const lines = escapeHtml(text).split(/\r?\n/);
    const htmlParts = [];
    let paragraphBuffer = [];
    let listBuffer = [];
    let listType = null;
    // Tracks where the next <ol> should resume counting from, so a numbered
    // section interrupted by a bullet sub-list (e.g. "1. Heading" / "- detail"
    // / "2. Heading") keeps counting up instead of every <ol> restarting at 1.
    let orderedStart = 1;

    const flushParagraph = () => {
        if (paragraphBuffer.length) {
            htmlParts.push(`<p>${paragraphBuffer.join('<br>')}</p>`);
            paragraphBuffer = [];
            orderedStart = 1;
        }
    };

    const flushList = () => {
        if (listBuffer.length) {
            const tag = listType;
            const startAttr = tag === 'ol' && orderedStart > 1 ? ` start="${orderedStart}"` : '';
            htmlParts.push(`<${tag}${startAttr}>${listBuffer.map((item) => `<li>${item}</li>`).join('')}</${tag}>`);
            if (tag === 'ol') {
                orderedStart += listBuffer.length;
            }
            listBuffer = [];
            listType = null;
        }
    };

    for (const rawLine of lines) {
        const line = rawLine.trim();

        if (!line) {
            flushParagraph();
            flushList();
            continue;
        }

        const headingMatch = line.match(/^\*\*(.+?)\*\*:?$/);
        if (headingMatch) {
            flushParagraph();
            flushList();
            htmlParts.push(`<h4>${headingMatch[1]}</h4>`);
            orderedStart = 1;
            continue;
        }

        const bulletMatch = line.match(/^[-*]\s+(.*)$/);
        const numberedMatch = line.match(/^\d+\.\s+(.*)$/);
        if (bulletMatch || numberedMatch) {
            const newType = bulletMatch ? 'ul' : 'ol';
            if (listType && listType !== newType) flushList();
            listType = newType;
            flushParagraph();
            listBuffer.push(applyInlineMarkdown((bulletMatch || numberedMatch)[1]));
            continue;
        }

        flushList();
        paragraphBuffer.push(applyInlineMarkdown(line));
    }

    flushParagraph();
    flushList();

    return htmlParts.join('\n');
}

// Needed for Prisma to connect to database
const { Pool } = require('pg');
const { PrismaPg } = require('@prisma/adapter-pg');
const { PrismaClient } = require('@prisma/client');
const pool = new Pool({ connectionString: process.env.DATABASE_URL });
const adapter = new PrismaPg(pool);
const prisma = new PrismaClient({ adapter });

// Main landing page
app.get('/', function(req, res) {
    res.render('pages/home', { result: null, error: null, amendedDocx: null, amendedDocxFilename: null });
});

// About landing page
app.get('/about', function(req, res) {
    res.render('pages/about');
});

// Handle AI input submission
app.post('/generate', upload.single('attachment'), async function(req, res) {
    try {
        const { userInput } = req.body;
        const file = req.file;

        if (!file) {
            return res.render('pages/home', {
                error: 'Please attach a file.',
                result: null,
                amendedDocx: null,
                amendedDocxFilename: null
            });
        }

        // Build form data to send to Python backend
        const formData = new FormData();
        formData.append('query', userInput || '');
        formData.append('file', fs.createReadStream(file.path), file.originalname);

        // Call Python backend
        const response = await fetch(process.env.PYTHON_BACKEND_URL + '/process', {
            method: 'POST',
            body: formData,
            headers: formData.getHeaders(),
        });

        const data = await response.json();

        // Clean up uploaded file after processing
        fs.unlinkSync(file.path);

        if (!response.ok || data.status === 'error') {
            return res.render('pages/home', {
                error: data.detail || data.message || 'Something went wrong.',
                result: null,
                amendedDocx: null,
                amendedDocxFilename: null
            });
        }

        res.render('pages/home', {
            result: formatResult(data.result),
            error: null,
            amendedDocx: data.amended_docx_base64 || null,
            amendedDocxFilename: data.amended_docx_base64 ? `amended-${file.originalname}` : null
        });
    } catch (error) {
        console.log(error);
        res.render('pages/home', {
            error: 'Something went wrong.',
            result: null,
            amendedDocx: null,
            amendedDocxFilename: null
        });
    }
});

// Tells the app which port to run on
const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0',() => {
    console.log(`Server running on port ${PORT}`);
});