/* Local YouTube URL validation and canonicalization. This module makes no network requests. */
(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.StudySphereYouTube = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
    const YOUTUBE_HOSTS = new Set([
        'youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com'
    ]);

    function canonicalizeYouTubeUrl(input) {
        if (typeof input !== 'string' || !input.trim()) {
            return { valid: false, error: 'Paste a YouTube lecture URL.' };
        }

        let parsed;
        try {
            parsed = new URL(input.trim());
        } catch (error) {
            return { valid: false, error: 'Enter a valid YouTube URL.' };
        }

        if (parsed.protocol !== 'https:' || parsed.username || parsed.password || parsed.port) {
            return { valid: false, error: 'Use a standard HTTPS YouTube link.' };
        }

        const host = parsed.hostname.toLowerCase();
        let videoId = null;
        if (host === 'youtu.be') {
            const match = parsed.pathname.match(/^\/([A-Za-z0-9_-]{11})\/?$/);
            videoId = match && match[1];
        } else if (YOUTUBE_HOSTS.has(host)) {
            if (parsed.pathname === '/watch' || parsed.pathname === '/') {
                const ids = parsed.searchParams.getAll('v');
                if (ids.length === 1) videoId = ids[0];
            } else {
                const match = parsed.pathname.match(/^\/(?:shorts|embed|live)\/([A-Za-z0-9_-]{11})\/?$/);
                videoId = match && match[1];
            }
        }

        if (!videoId || !VIDEO_ID.test(videoId)) {
            return { valid: false, error: 'This link is not a supported YouTube video URL.' };
        }

        return {
            valid: true,
            videoId: videoId,
            canonicalUrl: `https://www.youtube.com/watch?v=${videoId}`
        };
    }

    function restoreTranscriptSelection(lecture, transcriptSelect, materialFilter) {
        if (!lecture || !lecture.transcript_available || lecture.transcript_material_id === null
            || lecture.transcript_material_id === undefined) return false;
        const materialId = String(lecture.transcript_material_id);
        const hasMaterial = Array.from(transcriptSelect.options || [])
            .some(option => String(option.value) === materialId);
        if (!hasMaterial) return false;
        transcriptSelect.value = materialId;
        materialFilter.value = materialId;
        return true;
    }

    return { canonicalizeYouTubeUrl, restoreTranscriptSelection };
});
