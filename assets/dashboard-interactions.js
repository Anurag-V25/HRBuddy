/* Enhanced Dashboard Interactions - Enterprise HR Analytics Suite */

// Global variables
let sidebarOpen = false;
let chatbotOpen = false;
let refreshTimer = 30;
let refreshInterval;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    startRefreshTimer();
    setupEventListeners();
    hideLoadingOverlay();
});

// Initialize dashboard components
function initializeDashboard() {
    console.log('🚀 Initializing Enterprise HR Analytics Dashboard...');
    
    // Update last updated time
    updateLastUpdatedTime();
    
    // Initialize sidebar
    initializeSidebar();
    
    // Initialize chatbot
    initializeChatbot();
    
    // Initialize section navigation
    initializeSectionNavigation();
    
    // Add smooth scrolling
    addSmoothScrolling();
    
    // Initialize tooltips
    initializeTooltips();
    
    console.log('✅ Dashboard initialized successfully');
}

// Setup event listeners
function setupEventListeners() {
    // Window resize handler
    window.addEventListener('resize', handleWindowResize);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // Scroll handler for section highlighting
    window.addEventListener('scroll', handleScroll);
}

// Sidebar functionality
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('main-content');
    const toggleBtn = document.querySelector('.sidebar-toggle');
    
    if (sidebarOpen) {
        sidebar.classList.remove('open');
        mainContent.classList.remove('sidebar-open');
        toggleBtn.innerHTML = '<i class="fas fa-bars"></i>';
        sidebarOpen = false;
    } else {
        sidebar.classList.add('open');
        mainContent.classList.add('sidebar-open');
        toggleBtn.innerHTML = '<i class="fas fa-times"></i>';
        sidebarOpen = true;
    }
    
    // Add animation class
    sidebar.classList.add('slide-in-up');
    setTimeout(() => sidebar.classList.remove('slide-in-up'), 500);
}

function initializeSidebar() {
    const sidebarLinks = document.querySelectorAll('.sidebar-link');
    
    sidebarLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Remove active class from all links
            sidebarLinks.forEach(l => l.classList.remove('active'));
            
            // Add active class to clicked link
            this.classList.add('active');
            
            // Close sidebar on mobile
            if (window.innerWidth <= 768) {
                toggleSidebar();
            }
        });
    });
}

// Chatbot functionality
function toggleChatbot() {
    const chatbotModal = document.getElementById('chatbot-modal');
    const toggleBtn = document.getElementById('chatbot-toggle');
    const notificationBadge = document.querySelector('.notification-badge');
    
    if (chatbotOpen) {
        chatbotModal.style.display = 'none';
        toggleBtn.innerHTML = '<i class="fas fa-comments"></i><span class="notification-badge">1</span>';
        chatbotOpen = false;
    } else {
        chatbotModal.style.display = 'flex';
        toggleBtn.innerHTML = '<i class="fas fa-times"></i>';
        notificationBadge.style.display = 'none';
        chatbotOpen = true;
        
        // Focus on input field
        setTimeout(() => {
            document.getElementById('chatbot-input-field').focus();
        }, 100);
    }
}

function initializeChatbot() {
    const inputField = document.getElementById('chatbot-input-field');
    const messagesContainer = document.getElementById('chatbot-messages');
    
    // Auto-resize messages container
    function resizeMessagesContainer() {
        const headerHeight = document.querySelector('.chatbot-header').offsetHeight;
        const inputHeight = document.querySelector('.chatbot-input').offsetHeight;
        const totalHeight = 500 - headerHeight - inputHeight;
        messagesContainer.style.height = totalHeight + 'px';
    }
    
    resizeMessagesContainer();
    window.addEventListener('resize', resizeMessagesContainer);
}

function sendChatbotMessage() {
    const inputField = document.getElementById('chatbot-input-field');
    const message = inputField.value.trim();
    
    if (message) {
        addUserMessage(message);
        inputField.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Call chatbot API
        fetch('/api/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            addBotMessage(data.message, data.quick_replies);
        })
        .catch(error => {
            hideTypingIndicator();
            console.error('Error:', error);
            addBotMessage("Sorry, I'm having trouble processing your request right now. Please try again.");
        });
    }
}

function handleChatbotInput(event) {
    if (event.key === 'Enter') {
        sendChatbotMessage();
    }
}

function addUserMessage(message) {
    const messagesContainer = document.getElementById('chatbot-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.innerHTML = `
        <div class="message-content">
            ${message}
        </div>
    `;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom(messagesContainer);
}

function addBotMessage(message, quickReplies = null) {
    const messagesContainer = document.getElementById('chatbot-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    let messageContent = `
        <div class="message-content">
            <i class="fas fa-robot me-2"></i>
            ${message}
        </div>
    `;
    
    // Add quick reply buttons if provided
    if (quickReplies && quickReplies.length > 0) {
        messageContent += '<div class="quick-replies">';
        quickReplies.forEach(reply => {
            messageContent += `<button class="quick-reply-btn" onclick="handleQuickReply('${reply}')">${reply}</button>`;
        });
        messageContent += '</div>';
    }
    
    messageDiv.innerHTML = messageContent;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom(messagesContainer);
}

function scrollToBottom(element) {
    element.scrollTop = element.scrollHeight;
}

function showTypingIndicator() {
    const messagesContainer = document.getElementById('chatbot-messages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-content">
            <i class="fas fa-robot me-2"></i>
            <span class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </span>
                </div>
    `;
    messagesContainer.appendChild(typingDiv);
    scrollToBottom(messagesContainer);
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function handleQuickReply(reply) {
    const inputField = document.getElementById('chatbot-input-field');
    inputField.value = reply;
    sendChatbotMessage();
}

// Chatbot response logic - now handled by backend API

// Section navigation
function scrollToSection(sectionId) {
            const section = document.getElementById(sectionId);
            if (section) {
        const offset = 100; // Account for fixed navbar
        const targetPosition = section.offsetTop - offset;
        
        window.scrollTo({
            top: targetPosition,
            behavior: 'smooth'
        });
        
        // Update active section in sidebar
        updateActiveSection(sectionId);
    }
}

function initializeSectionNavigation() {
    const sections = ['executive-summary', 'hr-operations', 'strategic-planning', 'real-time', 'advanced-analytics'];
    
    // Create intersection observer for section highlighting
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                updateActiveSection(entry.target.id);
            }
        });
    }, { threshold: 0.3 });
    
    // Observe all sections
    sections.forEach(sectionId => {
        const section = document.getElementById(sectionId);
        if (section) {
            observer.observe(section);
        }
    });
}

function updateActiveSection(sectionId) {
    const sidebarLinks = document.querySelectorAll('.sidebar-link');
    
    sidebarLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${sectionId}`) {
            link.classList.add('active');
        }
    });
}

// Smooth scrolling
function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            scrollToSection(targetId);
        });
    });
}

// Refresh functionality
function refreshDashboard() {
    console.log('🔄 Refreshing dashboard...');
    
    // Show loading indicator
    showNotification('Refreshing data...', 'info');
    
    // Simulate refresh delay
    setTimeout(() => {
        updateLastUpdatedTime();
        showNotification('Dashboard refreshed successfully!', 'success');
        resetRefreshTimer();
    }, 2000);
}

function startRefreshTimer() {
    refreshInterval = setInterval(() => {
        refreshTimer--;
        updateRefreshTimer();
        
        if (refreshTimer <= 0) {
            refreshDashboard();
        }
    }, 1000);
}

function resetRefreshTimer() {
    refreshTimer = 30;
    updateRefreshTimer();
}

function updateRefreshTimer() {
    const timerElement = document.getElementById('refresh-timer');
    if (timerElement) {
        timerElement.textContent = `${refreshTimer}s`;
    }
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="fas fa-${getNotificationIcon(type)} me-2"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Show notification
    setTimeout(() => notification.classList.add('show'), 100);
    
    // Remove notification after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'check-circle';
        case 'warning': return 'exclamation-triangle';
        case 'error': return 'times-circle';
        default: return 'info-circle';
    }
}

// Utility functions
function updateLastUpdatedTime() {
    const timeElement = document.getElementById('last-updated-time');
    if (timeElement) {
        const now = new Date();
        timeElement.textContent = now.toLocaleTimeString();
    }
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        setTimeout(() => {
            overlay.style.opacity = '0';
            setTimeout(() => overlay.style.display = 'none', 500);
        }, 1000);
    }
}

function handleWindowResize() {
    // Close sidebar on mobile when window is resized
    if (window.innerWidth > 768 && sidebarOpen) {
        toggleSidebar();
    }
}

function handleKeyboardShortcuts(event) {
    // Ctrl/Cmd + R to refresh
    if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
        event.preventDefault();
        refreshDashboard();
    }
    
    // Ctrl/Cmd + / to toggle sidebar
    if ((event.ctrlKey || event.metaKey) && event.key === '/') {
        event.preventDefault();
        toggleSidebar();
    }
    
    // Ctrl/Cmd + K to toggle chatbot
    if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        toggleChatbot();
    }
}

function handleScroll() {
    // Add scroll-based animations
    const elements = document.querySelectorAll('.section-card');
    
    elements.forEach(element => {
        const rect = element.getBoundingClientRect();
        const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
        
        if (isVisible) {
            element.classList.add('fade-in');
        }
    });
}

// Export functionality
function exportDashboard(format) {
    console.log(`📊 Exporting dashboard as ${format}...`);
    
    showNotification(`Exporting dashboard as ${format.toUpperCase()}...`, 'info');
    
    // Simulate export process
        setTimeout(() => {
        showNotification(`Dashboard exported as ${format.toUpperCase()} successfully!`, 'success');
        }, 2000);
    }

// Alert functionality
function alertHighRisk() {
    showNotification('High-risk employees identified. Check the Risk Monitoring section.', 'warning');
    scrollToSection('real-time');
}

function openPlanningModal() {
    showNotification('Opening Workforce Planning tools...', 'info');
    // Add modal functionality here
}

// Tooltip initialization
function initializeTooltips() {
    // Initialize Bootstrap tooltips if available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    }

// Performance monitoring
function logPerformance() {
    if ('performance' in window) {
        const perfData = performance.getEntriesByType('navigation')[0];
        console.log('📈 Page Load Performance:', {
            'DOM Content Loaded': perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
            'Load Complete': perfData.loadEventEnd - perfData.loadEventStart,
            'Total Load Time': perfData.loadEventEnd - perfData.fetchStart
        });
    }
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('❌ Dashboard Error:', e.error);
    showNotification('Please wait the dashbaord is loading.', 'error');
});

// Unload cleanup
window.addEventListener('beforeunload', function() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
});

// Export functions for global access
window.toggleSidebar = toggleSidebar;
window.toggleChatbot = toggleChatbot;
window.sendChatbotMessage = sendChatbotMessage;
window.handleChatbotInput = handleChatbotInput;
window.scrollToSection = scrollToSection;
window.refreshDashboard = refreshDashboard;
window.exportDashboard = exportDashboard;
window.alertHighRisk = alertHighRisk;
window.openPlanningModal = openPlanningModal;

console.log('🎯 Dashboard interactions loaded successfully');
console.log('🎯 Dashboard interactions loaded successfully');