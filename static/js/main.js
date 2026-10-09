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

    // Chat input handling (visual only)
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');

    if (chatInput && sendButton && chatMessages) {
        const sendMessage = function() {
            const message = chatInput.value.trim();
            if (message) {
                // Visual feedback only - no fake AI responses
                console.log('Message sent (visual only - backend not connected):', message);
                chatInput.value = '';
            }
        };

        sendButton.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
});
