document.addEventListener('DOMContentLoaded', function() {
    console.log('StudySphere AI loaded');

    // Mobile sidebar toggle
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const sidebar = document.querySelector('.sidebar');

    if (mobileMenuToggle && sidebar) {
        mobileMenuToggle.addEventListener('click', function() {
            sidebar.classList.toggle('open');
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', function(e) {
            if (window.innerWidth <= 768) {
                if (!sidebar.contains(e.target) && !mobileMenuToggle.contains(e.target)) {
                    sidebar.classList.remove('open');
                }
            }
        });
    }

    // Set active navigation item
    const currentPath = window.location.pathname;
    const navItems = document.querySelectorAll('.nav-item');

    navItems.forEach(item => {
        const href = item.getAttribute('href');
        if (href && currentPath === href) {
            item.classList.add('active');
        }
    });

    // Upload area drag and drop handling
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const selectedFileName = document.getElementById('selectedFileName');
    const uploadForm = document.getElementById('materialUploadForm');
    const uploadButton = document.getElementById('materialUploadButton');
    const uploadProgress = document.getElementById('uploadProgress');

    if (uploadArea && fileInput) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--primary-color)';
            uploadArea.style.backgroundColor = 'var(--bg-tertiary)';
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--border-color)';
            uploadArea.style.backgroundColor = 'transparent';
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--border-color)';
            uploadArea.style.backgroundColor = 'transparent';

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                if (selectedFileName) {
                    selectedFileName.textContent = 'Selected: ' + files[0].name;
                }
            }
        });

        // Update selected filename when file is chosen
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0 && selectedFileName) {
                selectedFileName.textContent = 'Selected: ' + e.target.files[0].name;
            }
        });
    }

    if (uploadForm && fileInput && uploadButton && uploadProgress) {
        const uploadButtonLabel = uploadButton.textContent;

        function resetUploadState() {
            uploadButton.disabled = false;
            uploadButton.textContent = uploadButtonLabel;
            uploadProgress.style.display = 'none';
            uploadProgress.setAttribute('aria-hidden', 'true');
            uploadForm.removeAttribute('aria-busy');
        }

        uploadForm.addEventListener('submit', function(event) {
            // Let the server render its existing "No file selected" flash message.
            if (!fileInput.files || fileInput.files.length === 0) return;

            uploadButton.disabled = true;
            uploadButton.textContent = 'Uploading and processing...';
            uploadProgress.style.display = 'flex';
            uploadProgress.setAttribute('aria-hidden', 'false');
            uploadForm.setAttribute('aria-busy', 'true');

            // Restore the controls if another client-side handler cancels submission.
            window.setTimeout(function() {
                if (event.defaultPrevented) resetUploadState();
            }, 0);
        });

        // Also reset if the browser restores this page from its back/forward cache.
        window.addEventListener('pageshow', resetUploadState);
    }

    // Validate and open only known YouTube URL shapes; this performs no network request.
    const youtubeUrlInput = document.getElementById('youtubeLectureUrl');
    const youtubeOpenButton = document.getElementById('openYouTubeLecture');
    const youtubeStatus = document.getElementById('youtubeLectureStatus');
    const youtubePlayerLink = document.getElementById('youtubePlayerLink');
    const youtubeTranscriptSelect = document.getElementById('youtubeTranscriptMaterial');
    const youtubeLectureTitle = document.getElementById('youtubeLectureTitle');
    const saveYouTubeLecture = document.getElementById('saveYouTubeLecture');
    const savedYouTubeLecture = document.getElementById('savedYouTubeLecture');
    const materialFilter = document.getElementById('materialFilter');
    let validatedYouTubeUrl = null;

    function updateSaveLectureButton() {
        if (saveYouTubeLecture) {
            saveYouTubeLecture.disabled = !validatedYouTubeUrl
                || !youtubeTranscriptSelect || !youtubeTranscriptSelect.value;
        }
    }

    if (youtubeUrlInput && youtubeOpenButton && youtubeStatus && youtubePlayerLink
        && window.StudySphereYouTube) {
        youtubeUrlInput.addEventListener('input', function() {
            validatedYouTubeUrl = null;
            youtubePlayerLink.hidden = true;
            youtubePlayerLink.removeAttribute('href');
            youtubeStatus.textContent = '';
            updateSaveLectureButton();
        });

        youtubeOpenButton.addEventListener('click', function() {
            const result = window.StudySphereYouTube.canonicalizeYouTubeUrl(youtubeUrlInput.value);
            if (!result.valid) {
                validatedYouTubeUrl = null;
                youtubePlayerLink.hidden = true;
                youtubePlayerLink.removeAttribute('href');
                youtubeStatus.textContent = result.error;
                youtubeStatus.style.color = 'var(--danger-color)';
                updateSaveLectureButton();
                return;
            }

            validatedYouTubeUrl = result.canonicalUrl;
            youtubePlayerLink.href = result.canonicalUrl;
            youtubePlayerLink.hidden = false;
            youtubeStatus.textContent = `Valid YouTube link (${result.videoId}). Open the official player below.`;
            youtubeStatus.style.color = 'var(--text-secondary)';
            updateSaveLectureButton();
        });
    }

    if (youtubeTranscriptSelect && materialFilter) {
        youtubeTranscriptSelect.addEventListener('change', function() {
            materialFilter.value = youtubeTranscriptSelect.value;
            materialFilter.dispatchEvent(new Event('change', { bubbles: true }));
            updateSaveLectureButton();
        });
    }

    if (savedYouTubeLecture && youtubeUrlInput && youtubeTranscriptSelect
        && youtubeLectureTitle && youtubePlayerLink && youtubeStatus && materialFilter
        && window.StudySphereYouTube) {
        savedYouTubeLecture.addEventListener('change', function() {
            const option = savedYouTubeLecture.selectedOptions[0];
            if (!option || !option.value) return;
            const lecture = {
                canonical_url: option.dataset.url,
                display_title: option.dataset.title,
                transcript_material_id: option.dataset.transcriptId || null,
                transcript_available: option.dataset.transcriptAvailable === 'true'
            };
            youtubeUrlInput.value = lecture.canonical_url || '';
            youtubeLectureTitle.value = lecture.display_title || '';
            const selectionRestored = window.StudySphereYouTube.restoreTranscriptSelection(
                lecture, youtubeTranscriptSelect, materialFilter
            );
            if (!selectionRestored) {
                youtubeTranscriptSelect.value = '';
                materialFilter.value = '';
                materialFilter.dispatchEvent(new Event('change', { bubbles: true }));
            }
            youtubeTranscriptSelect.dispatchEvent(new Event('change', { bubbles: true }));
            const result = window.StudySphereYouTube.canonicalizeYouTubeUrl(lecture.canonical_url);
            validatedYouTubeUrl = result.valid ? result.canonicalUrl : null;
            youtubePlayerLink.href = validatedYouTubeUrl || '#';
            youtubePlayerLink.hidden = !validatedYouTubeUrl;
            youtubeStatus.textContent = selectionRestored
                ? 'Saved lecture and indexed transcript selected.'
                : 'This lecture’s transcript is unavailable. Upload or select an indexed timestamped transcript before saving a replacement association.';
            youtubeStatus.style.color = selectionRestored ? 'var(--text-secondary)' : 'var(--danger-color)';
            updateSaveLectureButton();
        });
    }

    if (saveYouTubeLecture && youtubeTranscriptSelect && youtubeLectureTitle
        && savedYouTubeLecture && youtubeStatus) {
        saveYouTubeLecture.addEventListener('click', async function() {
            if (!validatedYouTubeUrl || !youtubeTranscriptSelect.value) return;
            saveYouTubeLecture.disabled = true;
            youtubeStatus.textContent = 'Saving lecture and checking its indexed timestamped transcript…';
            try {
                const response = await fetch('/api/youtube-lectures', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
                    body: JSON.stringify({
                        url: validatedYouTubeUrl,
                        transcript_material_id: Number(youtubeTranscriptSelect.value),
                        display_title: youtubeLectureTitle.value.trim()
                    })
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'The lecture could not be saved.');
                const lecture = data.lecture;
                let option = Array.from(savedYouTubeLecture.options)
                    .find(item => item.value === lecture.video_id);
                if (!option) {
                    option = document.createElement('option');
                    option.value = lecture.video_id;
                    savedYouTubeLecture.appendChild(option);
                }
                option.dataset.url = lecture.canonical_url;
                option.dataset.title = lecture.display_title || '';
                option.dataset.transcriptId = String(lecture.transcript_material_id || '');
                option.dataset.transcriptAvailable = String(lecture.transcript_available);
                option.textContent = lecture.display_title || lecture.video_id;
                savedYouTubeLecture.value = lecture.video_id;
                savedYouTubeLecture.dispatchEvent(new Event('change', { bubbles: true }));
                youtubeStatus.textContent = 'Lecture and transcript association saved.';
                youtubeStatus.style.color = 'var(--success-color)';
            } catch (error) {
                youtubeStatus.textContent = error.message || 'The lecture could not be saved.';
                youtubeStatus.style.color = 'var(--danger-color)';
                updateSaveLectureButton();
            }
        });
    }

    // AI Tutor Semantic Search & Grounded Retrieval Handling
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const initialEmptyState = document.getElementById('initialEmptyState');

    if (chatInput && sendButton && chatMessages) {
        function removeInitialEmptyState() {
            if (initialEmptyState && initialEmptyState.parentNode) {
                initialEmptyState.parentNode.removeChild(initialEmptyState);
            }
        }

        function appendUserMessage(text) {
            removeInitialEmptyState();
            const msg = document.createElement('div');
            msg.className = 'chat-message user';

            const textElem = document.createElement('div');
            textElem.className = 'chat-message-text';
            textElem.textContent = text; // Safe text rendering

            msg.appendChild(textElem);
            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function appendLoadingMessage() {
            const msg = document.createElement('div');
            msg.className = 'chat-message assistant';
            msg.id = 'activeLoadingBubble';

            const loadingContainer = document.createElement('div');
            loadingContainer.className = 'loading-indicator';

            const spinner = document.createElement('div');
            spinner.className = 'spinner';

            const textSpan = document.createElement('span');
            textSpan.textContent = 'Searching course knowledge base for relevant passages...';

            loadingContainer.appendChild(spinner);
            loadingContainer.appendChild(textSpan);
            msg.appendChild(loadingContainer);

            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            return msg;
        }

        function appendAssistantError(errorText) {
            const msg = document.createElement('div');
            msg.className = 'chat-message assistant';

            const errorElem = document.createElement('div');
            errorElem.className = 'chat-error-message';
            errorElem.textContent = '⚠️ ' + errorText; // Safe text rendering

            msg.appendChild(errorElem);
            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function appendRetrievalResults(data) {
            const msg = document.createElement('div');
            msg.className = 'chat-message assistant';
            const results = data && Array.isArray(data.results)
                ? data.results.filter(item => item && typeof item === 'object' && !Array.isArray(item))
                : [];

            // 1. Header with title
            const header = document.createElement('div');
            header.className = 'retrieval-header';

            const title = document.createElement('div');
            title.className = 'retrieval-title';
            title.textContent = '📚 Retrieved Study Material';

            const countBadge = document.createElement('span');
            countBadge.className = 'badge badge-primary';
            const count = results.length;
            countBadge.textContent = `${count} match${count === 1 ? '' : 'es'}`;

            header.appendChild(title);
            header.appendChild(countBadge);
            msg.appendChild(header);

            // 2. Render passage cards
            if (results.length === 0) {
                const emptyMsg = document.createElement('p');
                emptyMsg.style.color = 'var(--text-secondary)';
                emptyMsg.style.fontSize = 'var(--font-size-sm)';
                emptyMsg.textContent = 'Your uploaded course materials did not provide a sufficiently relevant match for this question. Try rephrasing your query or selecting another material.';
                msg.appendChild(emptyMsg);
            } else {
                results.forEach((item) => {
                    const card = document.createElement('div');
                    card.className = 'passage-card';

                    // Citation header
                    const citation = document.createElement('div');
                    citation.className = 'passage-citation';

                    const source = document.createElement('span');
                    source.className = 'passage-source';
                    const fileText = item.source_file || 'Course Material';
                    const locationType = item.location_type === 'slide' ? 'Slide'
                        : item.location_type === 'timestamp' ? 'Timestamp' : 'Page';
                    const locationValue = item.location_type === 'timestamp'
                        ? item.location_value : (item.location_value ?? item.page_number);
                    const locationText = locationValue !== null && locationValue !== undefined
                        && String(locationValue).trim() !== ''
                        ? ` • ${locationType} ${locationValue}`
                        : '';
                    source.textContent = `📄 ${fileText}${locationText}`;

                    const scoreBadge = document.createElement('span');
                    scoreBadge.className = 'passage-score-badge';
                    const scorePct = Math.round((item.similarity_score || 0) * 100);
                    scoreBadge.textContent = `Match: ${scorePct}%`;

                    citation.appendChild(source);
                    citation.appendChild(scoreBadge);
                    card.appendChild(citation);

                    // Passage content
                    const textContent = document.createElement('div');
                    textContent.className = 'passage-text';
                    textContent.textContent = item.content || ''; // Safe text rendering
                    card.appendChild(textContent);

                    msg.appendChild(card);
                });
            }

            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function appendGeneratedAnswer(data) {
            const msg = document.createElement('div');
            msg.className = 'chat-message assistant';

            // Header
            const header = document.createElement('div');
            header.className = 'answer-header';

            const title = document.createElement('div');
            title.className = 'answer-title';
            title.textContent = data.insufficient_context
                ? '📚 Insufficient study material'
                : '🤖 AI Tutor Answer';

            header.appendChild(title);
            msg.appendChild(header);

            // Answer content
            if (data.insufficient_context) {
                const insufficientMsg = document.createElement('div');
                insufficientMsg.className = 'answer-insufficient';
                insufficientMsg.textContent = data.answer || 'The uploaded materials do not contain enough relevant information to answer this question.';
                msg.appendChild(insufficientMsg);
            } else if (data.answer) {
                const answerText = document.createElement('div');
                answerText.className = 'answer-text';
                answerText.textContent = data.answer; // Safe text rendering
                msg.appendChild(answerText);
            } else if (data.error) {
                const errorMsg = document.createElement('div');
                errorMsg.className = 'chat-error-message';
                errorMsg.textContent = '⚠️ ' + data.error;
                msg.appendChild(errorMsg);
            }

            // Preserve any useful source citations for both answers and insufficient-context results.
            if (data.citations && data.citations.length > 0) {
                const citationsDiv = document.createElement('div');
                citationsDiv.className = 'answer-citations';

                const citationsLabel = document.createElement('div');
                citationsLabel.className = 'citations-label';
                citationsLabel.textContent = 'Sources:';
                citationsDiv.appendChild(citationsLabel);

                data.citations.forEach((citation) => {
                    const citationBadge = document.createElement('span');
                    citationBadge.className = 'citation-badge';
                    const fileText = citation.source_file || 'Unknown';
                    const locationType = citation.location_type === 'slide' ? 'slide'
                        : citation.location_type === 'timestamp' ? 'timestamp' : 'page';
                    const locationValue = citation.location_type === 'timestamp'
                        ? citation.location_value : (citation.location_value ?? citation.page_number);
                    const locatorText = locationValue !== null && locationValue !== undefined
                        && String(locationValue).trim() !== ''
                        ? `, ${locationType} ${locationValue}`
                        : '';
                    citationBadge.textContent = `[${fileText}${locatorText}]`;
                    citationsDiv.appendChild(citationBadge);
                });

                msg.appendChild(citationsDiv);
            }

            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function handleSearch() {
            const query = chatInput.value.trim();
            if (!query) return;

            // Render user bubble and clear input
            appendUserMessage(query);
            chatInput.value = '';
            chatInput.disabled = true;
            sendButton.disabled = true;

            const loadingBubble = appendLoadingMessage();

            try {
                const payload = {
                    query: query,
                    top_k: 4
                };
                if (materialFilter && materialFilter.value) {
                    payload.material_id = parseInt(materialFilter.value, 10);
                }

                const answerResponse = await fetch('/api/answer', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                if (loadingBubble && loadingBubble.parentNode) {
                    loadingBubble.parentNode.removeChild(loadingBubble);
                }

                let answerData;
                try {
                    answerData = await answerResponse.json();
                } catch (parseError) {
                    answerData = null;
                }

                if (!answerData || typeof answerData !== 'object' || Array.isArray(answerData)) {
                    appendAssistantError('The answer service returned an invalid response. Please try again.');
                } else {
                    const results = Array.isArray(answerData.results) ? answerData.results : [];
                    appendRetrievalResults({ results: results });

                    if (answerResponse.ok) {
                        appendGeneratedAnswer(answerData);
                    } else if (results.some(item => item && typeof item === 'object' && !Array.isArray(item))) {
                        // Keep retrieved passages visible when answer generation fails.
                        const warningMsg = document.createElement('div');
                        warningMsg.className = 'chat-message assistant';
                        const warningText = document.createElement('div');
                        warningText.className = 'chat-error-message';
                        warningText.textContent = '⚠️ Answer generation unavailable: ' + (answerData.error || 'Unknown error');
                        warningMsg.appendChild(warningText);
                        chatMessages.appendChild(warningMsg);
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    } else {
                        appendAssistantError(answerData.error || `Answer error (${answerResponse.status})`);
                    }
                }
            } catch (err) {
                if (loadingBubble && loadingBubble.parentNode) {
                    loadingBubble.parentNode.removeChild(loadingBubble);
                }
                appendAssistantError('Network or connection error. Please try again.');
                console.error('Search/answer request error:', err);
            } finally {
                chatInput.disabled = false;
                sendButton.disabled = false;
                chatInput.focus();
            }
        }

        sendButton.addEventListener('click', handleSearch);
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleSearch();
            }
        });
    }

    // Source-grounded assessment creation, submission, and feedback.
    const assessmentForm = document.getElementById('assessmentForm');
    if (assessmentForm) {
        const status = document.getElementById('assessmentStatus');
        const questionsPanel = document.getElementById('assessmentQuestions');
        const resultsPanel = document.getElementById('assessmentResults');
        const generateButton = document.getElementById('generateAssessmentButton');

        function setAssessmentStatus(message, isError) {
            status.textContent = message;
            status.style.color = isError ? 'var(--danger-color)' : 'var(--text-secondary)';
        }

        function renderAssessment(assessment) {
            questionsPanel.replaceChildren();
            resultsPanel.replaceChildren();
            resultsPanel.style.display = 'none';
            const heading = document.createElement('h2');
            heading.textContent = `${assessment.topic} (${assessment.difficulty})`;
            questionsPanel.appendChild(heading);

            const answerForm = document.createElement('form');
            answerForm.id = 'assessmentAnswerForm';
            assessment.questions.forEach((question, questionIndex) => {
                const fieldset = document.createElement('fieldset');
                fieldset.className = 'card';
                fieldset.style.margin = 'var(--spacing-lg) 0';
                const legend = document.createElement('legend');
                legend.textContent = `${questionIndex + 1}. ${question.question_text}`;
                fieldset.appendChild(legend);

                question.options.forEach((option, optionIndex) => {
                    const label = document.createElement('label');
                    label.style.display = 'block';
                    label.style.padding = 'var(--spacing-sm) 0';
                    const radio = document.createElement('input');
                    radio.type = 'radio';
                    radio.name = `question-${question.id}`;
                    radio.value = String(optionIndex);
                    radio.required = true;
                    label.appendChild(radio);
                    label.appendChild(document.createTextNode(` ${option}`));
                    fieldset.appendChild(label);
                });
                answerForm.appendChild(fieldset);
            });

            const submitButton = document.createElement('button');
            submitButton.type = 'submit';
            submitButton.className = 'btn btn-primary';
            submitButton.textContent = 'Submit Answers';
            answerForm.appendChild(submitButton);
            answerForm.addEventListener('submit', async function(event) {
                event.preventDefault();
                const answers = assessment.questions.map((question) => {
                    const selected = answerForm.querySelector(`input[name="question-${question.id}"]:checked`);
                    return {
                        question_id: question.id,
                        selected_option: selected ? Number(selected.value) : null
                    };
                });
                if (answers.some(answer => answer.selected_option === null)) {
                    setAssessmentStatus('Select an answer for every question before submitting.', true);
                    return;
                }
                submitButton.disabled = true;
                submitButton.textContent = 'Scoring...';
                setAssessmentStatus('Submitting your answers…', false);
                try {
                    const response = await fetch(`/api/assessments/${assessment.id}/submit`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
                        body: JSON.stringify({answers})
                    });
                    const data = await response.json();
                    if (!response.ok) throw new Error(data.error || 'Could not submit this assessment.');
                    renderAssessmentResults(data);
                    answerForm.querySelectorAll('input').forEach(input => { input.disabled = true; });
                    submitButton.textContent = 'Submitted';
                    setAssessmentStatus('Assessment completed. Review your answers and sources below.', false);
                } catch (error) {
                    setAssessmentStatus(error.message || 'Assessment submission failed.', true);
                    submitButton.disabled = false;
                    submitButton.textContent = 'Submit Answers';
                }
            });
            questionsPanel.appendChild(answerForm);
            questionsPanel.style.display = 'block';
        }

        function renderAssessmentResults(data) {
            resultsPanel.replaceChildren();
            const heading = document.createElement('h2');
            heading.textContent = `Score: ${data.score} / ${data.total_questions} (${data.percentage}%)`;
            resultsPanel.appendChild(heading);
            data.results.forEach((result, index) => {
                const item = document.createElement('section');
                item.className = 'card';
                item.style.margin = 'var(--spacing-lg) 0';
                const question = document.createElement('h3');
                question.textContent = `${index + 1}. ${result.question_text}`;
                item.appendChild(question);
                const outcome = document.createElement('p');
                outcome.textContent = result.is_correct
                    ? `Correct: ${result.selected_answer}`
                    : `Your answer: ${result.selected_answer} · Correct answer: ${result.correct_answer}`;
                outcome.style.color = result.is_correct ? 'var(--success-color)' : 'var(--danger-color)';
                item.appendChild(outcome);
                const explanation = document.createElement('p');
                explanation.textContent = result.explanation;
                item.appendChild(explanation);
                const citation = document.createElement('p');
                citation.textContent = `Source: ${result.citation}`;
                item.appendChild(citation);
                resultsPanel.appendChild(item);
            });
            resultsPanel.style.display = 'block';
        }

        assessmentForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            const payload = {
                topic: document.getElementById('assessmentTopic').value.trim(),
                difficulty: document.getElementById('assessmentDifficulty').value,
                question_count: Number(document.getElementById('assessmentCount').value)
            };
            const material = document.getElementById('assessmentMaterial').value;
            if (material) payload.material_id = Number(material);

            generateButton.disabled = true;
            generateButton.textContent = 'Generating…';
            questionsPanel.style.display = 'none';
            resultsPanel.style.display = 'none';
            setAssessmentStatus('Searching indexed materials and preparing grounded questions…', false);
            try {
                const response = await fetch('/api/assessments', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Could not generate the assessment.');
                renderAssessment(data.assessment);
                setAssessmentStatus('Questions are grounded in retrieved study passages. Correct answers are hidden until submission.', false);
            } catch (error) {
                setAssessmentStatus(error.message || 'Assessment generation failed.', true);
            } finally {
                generateButton.disabled = false;
                generateButton.textContent = 'Generate Quiz';
            }
        });
    }
});
