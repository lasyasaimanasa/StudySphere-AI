const assert = require('node:assert/strict');
const { canonicalizeYouTubeUrl, restoreTranscriptSelection } = require('../static/js/youtube_url.js');

const videoId = 'dQw4w9WgXcQ';
const canonical = `https://www.youtube.com/watch?v=${videoId}`;

for (const input of [
    canonical,
    `https://youtu.be/${videoId}?si=share-token`,
    `https://m.youtube.com/watch?v=${videoId}&list=PL123`,
    `https://www.youtube.com/shorts/${videoId}`,
    `https://www.youtube.com/embed/${videoId}`,
    `https://music.youtube.com/watch?v=${videoId}`,
    `https://www.youtube.com/live/${videoId}/`
]) {
    const result = canonicalizeYouTubeUrl(input);
    assert.equal(result.valid, true, input);
    assert.equal(result.videoId, videoId);
    assert.equal(result.canonicalUrl, canonical);
}

for (const input of [
    '',
    'not a url',
    `https://youtube.com.evil.example/watch?v=${videoId}`,
    `https://example.com/watch?v=${videoId}`,
    `http://youtube.com/watch?v=${videoId}`,
    `https://youtube.com/watch?v=${videoId.slice(1)}`,
    'https://youtube.com/watch?v=not-an-id',
    `https://youtube.com/watch?v=${videoId}&v=${videoId}`,
    `https://youtube.com/channel/${videoId}`,
    `https://user:pass@youtube.com/watch?v=${videoId}`
]) {
    assert.equal(canonicalizeYouTubeUrl(input).valid, false, input);
}

console.log('YouTube URL validation cases passed.');

const transcriptSelect = {options: [{value: ''}, {value: '21'}], value: ''};
const materialFilter = {value: ''};
assert.equal(restoreTranscriptSelection({
    transcript_available: true,
    transcript_material_id: 21
}, transcriptSelect, materialFilter), true);
assert.equal(transcriptSelect.value, '21');
assert.equal(materialFilter.value, '21');
assert.equal(restoreTranscriptSelection({
    transcript_available: false,
    transcript_material_id: null
}, transcriptSelect, materialFilter), false);
assert.equal(restoreTranscriptSelection({
    transcript_available: true,
    transcript_material_id: 999
}, transcriptSelect, materialFilter), false);
console.log('Saved transcript restoration cases passed.');
