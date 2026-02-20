// Clerk Authentication Modal Functions
function openClerkSignIn() {
    const container = document.getElementById('clerk-auth-container');
    const overlay = document.getElementById('clerk-overlay');

    if (!container || !overlay) {
        console.error('Clerk auth container or overlay not found');
        return;
    }

    // Show the modal
    container.style.display = 'block';
    overlay.style.display = 'block';

    // Wait for Clerk to be ready
    const checkClerk = setInterval(() => {
        if (window.Clerk && Clerk.loaded) {
            clearInterval(checkClerk);
            Clerk.mountSignIn(container);
        }
    }, 100);

    // Stop checking after 10 seconds
    setTimeout(() => clearInterval(checkClerk), 10000);
}

function openClerkSignUp() {
    const container = document.getElementById('clerk-auth-container');
    const overlay = document.getElementById('clerk-overlay');

    if (!container || !overlay) {
        console.error('Clerk auth container or overlay not found');
        return;
    }

    // Show the modal
    container.style.display = 'block';
    overlay.style.display = 'block';

    // Wait for Clerk to be ready
    const checkClerk = setInterval(() => {
        if (window.Clerk && Clerk.loaded) {
            clearInterval(checkClerk);
            Clerk.mountSignUp(container);
        }
    }, 100);

    // Stop checking after 10 seconds
    setTimeout(() => clearInterval(checkClerk), 10000);
}

function closeClerkAuth() {
    const container = document.getElementById('clerk-auth-container');
    const overlay = document.getElementById('clerk-overlay');

    if (container) {
        container.style.display = 'none';
        container.innerHTML = ''; // Clear the mounted component
    }
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// Close modal when clicking overlay
document.addEventListener('DOMContentLoaded', () => {
    const overlay = document.getElementById('clerk-overlay');
    if (overlay) {
        overlay.addEventListener('click', closeClerkAuth);
    }
});

// Section switching functionality
function showSection(sectionName) {
    // Hide all sections
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.classList.remove('active');
    });

    // Remove active class from all nav pills
    const navPills = document.querySelectorAll('.nav-pill');
    navPills.forEach(pill => {
        pill.classList.remove('active');
    });

    // Also handle old nav-item class if exists
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.classList.remove('active');
    });

    // Show the selected section
    const targetSection = document.getElementById(sectionName + '-section');
    if (targetSection) {
        targetSection.classList.add('active');
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Mark the corresponding nav pill as active
    const targetNavPill = document.querySelector(`.nav-pill[data-section="${sectionName}"]`);
    if (targetNavPill) {
        targetNavPill.classList.add('active');
    }

    // Mark the corresponding nav item as active (legacy support)
    const targetNavItem = document.querySelector(`.nav-item[data-section="${sectionName}"]`);
    if (targetNavItem) {
        targetNavItem.classList.add('active');
    }

    // Update localStorage to remember current section
    localStorage.setItem('currentSection', sectionName);

    // Load user data if profile section
    if (sectionName === 'profile') {
        loadUserProfile();
    }
}

// Load user profile data from Clerk
function loadUserProfile() {
    // Wait for Clerk to be available
    const checkClerk = setInterval(() => {
        if (window.Clerk && Clerk.user) {
            clearInterval(checkClerk);

            const user = Clerk.user;
            const userName = document.getElementById('user-name');
            const userEmail = document.getElementById('user-email');

            if (userName) {
                // Display full name or first name, fallback to username or email
                const displayName = user.fullName || user.firstName || user.username || user.primaryEmailAddress?.emailAddress || 'User';
                userName.textContent = displayName;
            }

            if (userEmail) {
                // Display primary email address
                const email = user.primaryEmailAddress?.emailAddress || '';
                userEmail.textContent = email;
            }
        }
    }, 100); // Check every 100ms

    // Stop checking after 10 seconds
    setTimeout(() => {
        clearInterval(checkClerk);
        const userName = document.getElementById('user-name');
        if (userName && userName.textContent === 'Loading...') {
            userName.textContent = 'User';
        }
    }, 10000);
}

// Handle logout
async function handleLogout() {
    try {
        if (window.Clerk) {
            await Clerk.signOut();
            // Redirect to home/login page after logout
            window.location.href = '/';
        } else {
            console.error('Clerk is not loaded');
            alert('Unable to logout. Please refresh the page and try again.');
        }
    } catch (error) {
        console.error('Logout error:', error);
        alert('Failed to logout. Please try again.');
    }
}

// Handle form submissions via AJAX to stay on same page
function handleFormSubmit(event, form) {
    event.preventDefault();

    const formData = new FormData(form);
    const action = form.getAttribute('action');
    const method = form.getAttribute('method') || 'POST';

    fetch(action, {
        method: method,
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                console.warn('Form submission failed, falling back to traditional submit');
                form.submit();
                return null;
            }
            return response.text();
        })
        .then(html => {
            if (!html) return;
            // Create a temporary container to parse the HTML
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');

            // Check if there's a result container in the response
            const resultContent = doc.querySelector('.main-content') || doc.querySelector('.container') || doc.querySelector('.result-page-container') || doc.querySelector('.compare-result-container');

            if (resultContent) {
                // Create a modal or result overlay to show results
                showResultModal(resultContent.innerHTML);
            } else {
                // Fallback: If no container found, redirect to the response URL if possible or submit form
                console.warn('No result container found in response, falling back');
                form.submit();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            // Fallback: submit form normally
            form.submit();
        });

    return false;
}

// Show results in a modal
function showResultModal(content) {
    // Remove existing modal if any
    const existingModal = document.querySelector('.result-modal');
    if (existingModal) {
        existingModal.remove();
    }

    // Create modal
    const modal = document.createElement('div');
    modal.className = 'result-modal';
    modal.innerHTML = `
        <div class="result-modal-overlay" onclick="closeResultModal()"></div>
        <div class="result-modal-content">
            <button class="result-modal-close" onclick="closeResultModal()">
                <i class="fas fa-times"></i>
            </button>
            <div class="result-modal-body">
                <div class="modal-header-section">
                    <h2><i class="fas fa-futbol"></i> Player Selection Result</h2>
                    <div class="header-accent-line"></div>
                </div>
                <div class="modal-result-container">
                    ${content}
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // Animate modal in
    setTimeout(() => {
        modal.classList.add('active');
    }, 10);
}

// Close result modal
function closeResultModal() {
    const modal = document.querySelector('.result-modal');
    if (modal) {
        modal.classList.remove('active');
        setTimeout(() => {
            modal.remove();
        }, 300);
    }
}

// Theme Toggle Logic
document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('theme-toggle');
    const html = document.documentElement;
    const icon = toggleBtn ? toggleBtn.querySelector('i') : null;

    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    html.setAttribute('data-theme', savedTheme);
    updateIcon(savedTheme);

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateIcon(newTheme);
        });
    }

    function updateIcon(theme) {
        if (!icon) return;
        if (theme === 'light') {
            icon.className = 'fas fa-sun';
            icon.style.color = '#f59e0b'; // Sun color
        } else {
            icon.className = 'fas fa-moon';
            icon.style.color = 'white';
        }
    }
});

// Restore last viewed section functionality removed to ensure consistent redirect to home page as requested
document.addEventListener('DOMContentLoaded', function () {
    // Hamburger menu functionality
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');

    if (hamburger) {
        hamburger.addEventListener('click', function () {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    // Close mobile menu when clicking a nav link
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function () {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });

    // File upload preview for multiplayer section
    const fileInput = document.getElementById('file');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const removeFileBtn = document.getElementById('removeFile');
    const submitBtn = document.getElementById('submitBtn');
    const dropZone = document.getElementById('dropZone');

    if (fileInput) {
        fileInput.addEventListener('change', function () {
            if (this.files.length > 0) {
                fileName.textContent = this.files[0].name;
                fileInfo.style.display = 'flex';
                submitBtn.disabled = false;
            }
        });
    }

    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', function () {
            fileInput.value = '';
            fileInfo.style.display = 'none';
            submitBtn.disabled = true;
        });
    }

    // Drag and drop functionality
    if (dropZone) {
        dropZone.addEventListener('dragover', function (e) {
            e.preventDefault();
            this.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', function () {
            this.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', function (e) {
            e.preventDefault();
            this.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].name.endsWith('.csv')) {
                fileInput.files = files;
                fileName.textContent = files[0].name;
                fileInfo.style.display = 'flex';
                submitBtn.disabled = false;
            }
        });
    }

    // Animate probability bars on result page if they exist
    const probabilityFills = document.querySelectorAll('.probability-fill, .table-probability-fill');
    if (probabilityFills.length > 0) {
        probabilityFills.forEach(fill => {
            const width = fill.style.width;
            fill.style.width = '0';

            setTimeout(() => {
                fill.style.transition = 'width 1s ease-in-out';
                fill.style.width = width;
            }, 300);
        });
    }

    // Update usage stats from localStorage
    updateUsageStats();
});

// Update usage statistics
function updateUsageStats() {
    const singleCount = parseInt(localStorage.getItem('singlePlayerCount') || '0');
    const compareCount = parseInt(localStorage.getItem('compareCount') || '0');
    const multiCount = parseInt(localStorage.getItem('multiplayerCount') || '0');

    // Update profile stats
    const evalCountEl = document.getElementById('evaluationsCount');
    const compCountEl = document.getElementById('comparisonsCount');
    const uploadCountEl = document.getElementById('uploadsCount');

    if (evalCountEl) evalCountEl.textContent = singleCount;
    if (compCountEl) compCountEl.textContent = compareCount;
    if (uploadCountEl) uploadCountEl.textContent = multiCount;

    // Update usage bars
    const total = singleCount + compareCount + multiCount || 1;
    const singleFill = document.querySelector('[data-feature="single"]');
    const multiFill = document.querySelector('[data-feature="multi"]');
    const compareFill = document.querySelector('[data-feature="compare"]');

    if (singleFill) singleFill.style.width = (singleCount / total * 100) + '%';
    if (multiFill) multiFill.style.width = (multiCount / total * 100) + '%';
    if (compareFill) compareFill.style.width = (compareCount / total * 100) + '%';
}

// Increment usage counters
function incrementUsage(type) {
    const key = type + 'Count';
    const current = parseInt(localStorage.getItem(key) || '0');
    localStorage.setItem(key, current + 1);
    updateUsageStats();
}

// Function to handle tab switching (legacy support)
function openTab(tabName) {
    // Hide all tab contents
    const tabContents = document.getElementsByClassName('tab-content');
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }

    // Remove active class from all tab buttons
    const tabButtons = document.getElementsByClassName('tab-btn');
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }

    // Show the selected tab content and mark the button as active
    const targetTab = document.getElementById(tabName);
    if (targetTab) {
        targetTab.classList.add('active');
    }

    // Find the button that corresponds to this tab and mark it as active
    const buttons = document.getElementsByClassName('tab-btn');
    for (let i = 0; i < buttons.length; i++) {
        if (buttons[i].getAttribute('onclick') && buttons[i].getAttribute('onclick').includes(tabName)) {
            buttons[i].classList.add('active');
        }
    }
}
