// /app/assets/dashboard-animations.js

/**
 * DASHBOARD ANIMATIONS & VISUAL ENHANCEMENTS
 * ==========================================
 */

class DashboardAnimations {
    constructor() {
        this.animationQueue = [];
        this.isAnimating = false;
        this.observers = new Map();
        this.initializeAnimations();
    }

    initializeAnimations() {
        this.setupIntersectionObservers();
        this.setupCounterAnimations();
        this.setupChartAnimations();
        this.setupHoverEffects();
        this.setupLoadingStates();
    }

    setupIntersectionObservers() {
        // Fade in animation for sections
        const fadeInObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    this.animateCounters(entry.target);
                    fadeInObserver.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '50px'
        });

        // Observe all dashboard sections
        document.querySelectorAll('.dashboard-section').forEach(section => {
            fadeInObserver.observe(section);
        });

        // Slide in animation for chart cards
        const slideInObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateChartCard(entry.target);
                    slideInObserver.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.2
        });

        document.querySelectorAll('.chart-card').forEach(card => {
            slideInObserver.observe(card);
        });

        this.observers.set('fadeIn', fadeInObserver);
        this.observers.set('slideIn', slideInObserver);
    }

    animateChartCard(card) {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        
        // Random delay for staggered animation
        const delay = Math.random() * 300;
        
        setTimeout(() => {
            card.style.transition = 'all 0.8s cubic-bezier(0.22, 1, 0.36, 1)';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, delay);
    }

    setupCounterAnimations() {
        this.counterAnimationDuration = 2000; // 2 seconds
    }

    animateCounters(container) {
        const counters = container.querySelectorAll('.kpi-value');
        
        counters.forEach((counter, index) => {
            const text = counter.textContent;
            const numMatch = text.match(/[\d,]+\.?\d*/);
            
            if (numMatch) {
                const finalValue = parseFloat(numMatch[0].replace(/,/g, ''));
                const prefix = text.substring(0, text.indexOf(numMatch[0]));
                const suffix = text.substring(text.indexOf(numMatch[0]) + numMatch[0].length);
                
                // Animate with delay for staggered effect
                setTimeout(() => {
                    this.animateNumber(counter, 0, finalValue, prefix, suffix);
                }, index * 200);
            }
        });
    }

    animateNumber(element, start, end, prefix = '', suffix = '') {
        const duration = this.counterAnimationDuration;
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function for smooth animation
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const currentValue = start + (end - start) * easeOutQuart;
            
            // Format number based on original format
            let formattedValue;
            if (end >= 1000) {
                formattedValue = Math.round(currentValue).toLocaleString();
            } else if (end % 1 !== 0) {
                formattedValue = currentValue.toFixed(1);
            } else {
                formattedValue = Math.round(currentValue);
            }
            
            element.textContent = `${prefix}${formattedValue}${suffix}`;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }

    setupChartAnimations() {
        // Add entrance animations to charts
        this.chartAnimationConfig = {
            transition: {
                duration: 1000,
                easing: 'cubic-in-out'
            },
            frame: {
                duration: 1000
            }
        };
    }

    animateChartEntrance(chartId, delay = 0) {
        setTimeout(() => {
            const chartElement = document.getElementById(chartId);
            if (chartElement && chartElement.querySelector('.plotly-graph-div')) {
                const plotlyDiv = chartElement.querySelector('.plotly-graph-div');
                
                // Initial state
                plotlyDiv.style.opacity = '0';
                plotlyDiv.style.transform = 'scale(0.8)';
                
                // Animate in
                setTimeout(() => {
                    plotlyDiv.style.transition = 'all 0.8s cubic-bezier(0.22, 1, 0.36, 1)';
                    plotlyDiv.style.opacity = '1';
                    plotlyDiv.style.transform = 'scale(1)';
                }, 100);
            }
        }, delay);
    }

    setupHoverEffects() {
        // Enhanced hover effects for interactive elements
        document.addEventListener('mouseover', (e) => {
            if (e.target.closest('.chart-card')) {
                this.enhanceCardHover(e.target.closest('.chart-card'), true);
            }
            
            if (e.target.closest('.kpi-card')) {
                this.enhanceKPIHover(e.target.closest('.kpi-card'), true);
            }
            
            if (e.target.closest('.sidebar-link')) {
                this.enhanceLinkHover(e.target.closest('.sidebar-link'), true);
            }
        });

        document.addEventListener('mouseout', (e) => {
            if (e.target.closest('.chart-card')) {
                this.enhanceCardHover(e.target.closest('.chart-card'), false);
            }
            
            if (e.target.closest('.kpi-card')) {
                this.enhanceKPIHover(e.target.closest('.kpi-card'), false);
            }
            
            if (e.target.closest('.sidebar-link')) {
                this.enhanceLinkHover(e.target.closest('.sidebar-link'), false);
            }
        });
    }

    enhanceCardHover(card, isHovering) {
        const header = card.querySelector('.chart-header');
        const icon = card.querySelector('.chart-title i');
        
        if (isHovering) {
            card.style.transform = 'translateY(-8px) scale(1.02)';
            if (header) {
                header.style.background = 'rgba(37, 99, 235, 0.05)';
            }
            if (icon) {
                icon.style.transform = 'scale(1.2) rotate(5deg)';
                icon.style.color = '#2563eb';
            }
        } else {
            card.style.transform = '';
            if (header) {
                header.style.background = '';
            }
            if (icon) {
                icon.style.transform = '';
                icon.style.color = '';
            }
        }
    }

    enhanceKPIHover(kpi, isHovering) {
        const value = kpi.querySelector('.kpi-value');
        const label = kpi.querySelector('.kpi-label');
        
        if (isHovering) {
            kpi.style.transform = 'translateY(-4px) scale(1.05)';
            if (value) {
                value.style.transform = 'scale(1.1)';
            }
            if (label) {
                label.style.color = '#2563eb';
                label.style.fontWeight = '700';
            }
        } else {
            kpi.style.transform = '';
            if (value) {
                value.style.transform = '';
            }
            if (label) {
                label.style.color = '';
                label.style.fontWeight = '';
            }
        }
    }

    enhanceLinkHover(link, isHovering) {
        const icon = link.querySelector('i');
        
        if (isHovering) {
            if (icon) {
                icon.style.transform = 'translateX(5px) scale(1.1)';
            }
            link.style.paddingLeft = '1.75rem';
        } else {
            if (icon) {
                icon.style.transform = '';
            }
            link.style.paddingLeft = '';
        }
    }

    setupLoadingStates() {
        this.createLoadingAnimations();
    }

    createLoadingAnimations() {
        // Skeleton loading animation
        const style = document.createElement('style');
        style.textContent = `
            .skeleton-loading {
                background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                background-size: 200% 100%;
                animation: skeleton-loading 1.5s infinite;
            }
            
            @keyframes skeleton-loading {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
            
            .pulse-loading {
                animation: pulse-loading 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            }
            
            @keyframes pulse-loading {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .bounce-loading {
                animation: bounce-loading 1s infinite;
            }
            
            @keyframes bounce-loading {
                0%, 20%, 53%, 80%, 100% { transform: translate3d(0,0,0); }
                40%, 43% { transform: translate3d(0, -30px, 0); }
                70% { transform: translate3d(0, -15px, 0); }
                90% { transform: translate3d(0, -4px, 0); }
            }
        `;
        document.head.appendChild(style);
    }

    showChartLoading(chartId) {
        const container = document.getElementById(chartId);
        if (container) {
            const loadingHTML = `
                <div class="d-flex justify-content-center align-items-center" style="height: 300px;">
                    <div class="text-center">
                        <div class="spinner-border text-primary mb-3" role="status"></div>
                        <p class="text-muted">Loading chart data...</p>
                    </div>
                </div>
            `;
            container.innerHTML = loadingHTML;
        }
    }

    hideChartLoading(chartId) {
        const container = document.getElementById(chartId);
        if (container) {
            const loadingElement = container.querySelector('.spinner-border');
            if (loadingElement) {
                loadingElement.closest('.d-flex').remove();
            }
        }
    }

    // Page transition animations
    animatePageTransition(direction = 'in') {
        const mainContent = document.getElementById('main-content');
        if (!mainContent) return;

        if (direction === 'out') {
            mainContent.style.opacity = '0';
            mainContent.style.transform = 'translateY(20px)';
        } else {
            mainContent.style.transition = 'all 0.5s ease';
            mainContent.style.opacity = '1';
            mainContent.style.transform = 'translateY(0)';
        }
    }

    // Real-time data update animations
    animateDataUpdate(element, newValue) {
        if (!element) return;

        // Highlight animation for updated values
        element.style.transition = 'all 0.3s ease';
        element.style.backgroundColor = '#fef3c7';
        element.style.transform = 'scale(1.1)';

        setTimeout(() => {
            element.textContent = newValue;
            element.style.backgroundColor = '#dcfce7';
        }, 150);

        setTimeout(() => {
            element.style.backgroundColor = '';
            element.style.transform = 'scale(1)';
        }, 600);
    }

    // Chart update animations
    animateChartUpdate(chartId, newData) {
        const chartElement = document.getElementById(chartId);
        if (!chartElement) return;

        // Pulse animation before update
        chartElement.style.animation = 'pulse-loading 0.5s ease';

        setTimeout(() => {
            // Update chart data here
            chartElement.style.animation = '';
            
            // Success highlight
            chartElement.style.boxShadow = '0 0 20px rgba(34, 197, 94, 0.3)';
            setTimeout(() => {
                chartElement.style.boxShadow = '';
            }, 1000);
        }, 500);
    }

    // Notification animations
    slideInNotification(element) {
        element.style.transform = 'translateX(100%)';
        element.style.transition = 'transform 0.3s cubic-bezier(0.22, 1, 0.36, 1)';
        
        requestAnimationFrame(() => {
            element.style.transform = 'translateX(0)';
        });
    }

    slideOutNotification(element) {
        element.style.transform = 'translateX(100%)';
        element.style.opacity = '0';
        
        setTimeout(() => {
            element.remove();
        }, 300);
    }

    // Cleanup method
    destroy() {
        this.observers.forEach(observer => observer.disconnect());
        this.observers.clear();
        this.animationQueue = [];
    }
}

// Initialize animations when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardAnimations = new DashboardAnimations();
    
    // Expose utility functions globally
    window.animateChartEntrance = (chartId, delay) => {
        window.dashboardAnimations.animateChartEntrance(chartId, delay);
    };
    
    window.showChartLoading = (chartId) => {
        window.dashboardAnimations.showChartLoading(chartId);
    };
    
    window.hideChartLoading = (chartId) => {
        window.dashboardAnimations.hideChartLoading(chartId);
    };
});