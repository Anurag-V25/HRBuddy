// /app/assets/dashboard-core.js

/**
 * PRODUCTION-READY HR DASHBOARD - CORE FUNCTIONALITY
 * =================================================
 */

class HRDashboard {
    constructor() {
        this.initializeState();
        this.initializeEventListeners();
        this.loadDashboardData();
    // ...existing code...

    initializeState() {
        this.state = {
            sidebarOpen: window.innerWidth > 1200,
            currentSection: 'executive',
            refreshInterval: 30000, // 30 seconds
            lastUpdated: new Date(),
            dashboardData: null,
            charts: {},
            realTimeEnabled: true
        };
        
        // Store chart configurations
        this.chartConfigs = {
            theme: {
                layout: {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: { family: 'Inter, sans-serif', color: '#374151' },
                    margin: { t: 20, r: 20, b: 40, l: 40 }
                },
                mode: 'lines+markers'
            }
        };
                        const layout = { 
                            ...this.chartConfigs.theme.layout, 
                            title: '',
                            xaxis: { title: 'Job Role', tickangle: -45 },
                            yaxis: { title: 'Average Salary' }
                        };
                        return Plotly.newPlot('compensation-analytics-chart', data, layout, this.chartConfigs);
                    }
                }

                // Global utility functions for template interactions
    // ...existing code...

    // ...existing code...
    // ...existing code...

    generateTableData() {
        const roles = ['Data Scientist', 'Product Manager', 'Software Engineer'];
        return roles.map((role, i) => ({
            employeeId: 1000 + i,
            jobRole: role,
            department: i < 2 ? 'Tech' : 'Product',
            riskScore: (Math.random() * 0.3 + 0.7).toFixed(3)
        }));
    }

    generateFunnelData(levels) {
        const values = levels.map((_, i) => 100 - i * 20);
        return {
            y: levels,
            x: values,
            type: 'funnel'
        };
    }

    generateBarWithErrorData(categories) {
        return {
            x: categories,
            y: categories.map(() => Math.random() * 50000 + 60000),
            error_y: {
                type: 'data',
                array: categories.map(() => Math.random() * 10000 + 5000)
            },
            type: 'bar'
        };
    }

    async renderAllCharts() {
        if (!this.state.dashboardData?.charts) return;

        const chartPromises = [
            this.renderRiskHeatmap(),
            this.renderWorkforceROI(),
            this.renderAttritionForecast(),
            this.renderAttritionAnalysis(),
            this.renderRecruitmentPerformance(),
            this.renderEmployeeEngagement(),
            this.renderPerformanceAnalytics(),
            this.renderWorkforceDemographics(),
            this.renderCompensationIntelligence(),
            this.renderLearningROI(),
            this.renderManagerPerformance(),
            this.renderTalentPipeline(),
            this.renderJourneyMapping(),
            this.renderCompensationAnalytics(),
            this.renderWorkforcePlanning()
        ];

        await Promise.all(chartPromises);
        this.renderRiskMonitoringTable();
    }

    // Individual chart rendering methods
    renderRiskHeatmap() {
        const data = [this.state.dashboardData.charts.riskHeatmap];
        const layout = { ...this.chartConfigs.theme.layout, title: '' };
        return Plotly.newPlot('compensation-analytics-chart', data, layout, this.chartConfigs);
    }
}

    renderWorkforcePlanning() {
        const data = [this.state.dashboardData.charts.workforcePlanning];
        const layout = { 
            ...this.chartConfigs.theme.layout, 
            title: '',
            xaxis: { title: 'Month' },
            yaxis: { title: 'Projected Headcount' }
        };
        return Plotly.newPlot('workforce-planning-chart', data, layout, this.chartConfigs);
    }

    renderRiskMonitoringTable() {
        const tableData = this.state.dashboardData.charts.riskMonitoring;
        const tableHTML = this.generateTableHTML(tableData);
        document.getElementById('risk-monitoring-table').innerHTML = tableHTML;
    }

    generateTableHTML(data) {
        if (!data || data.length === 0) return '<p class="text-muted">No data available</p>';

        const headers = ['Employee ID', 'Job Role', 'Department', 'Risk Score'];
        const headerRow = headers.map(h => `<th>${h}</th>`).join('');
        
        const rows = data.map(row => `
            <tr class="${parseFloat(row.riskScore) > 0.8 ? 'table-danger' : ''}">
                <td>${row.employeeId}</td>
                <td>${row.jobRole}</td>
                <td>${row.department}</td>
                <td>
                    <span class="badge ${parseFloat(row.riskScore) > 0.8 ? 'bg-danger' : 'bg-warning'}">
                        ${row.riskScore}
                    </span>
                </td>
            </tr>
        `).join('');

        return `
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead><tr>${headerRow}</tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    }

    updateKPIs() {
        if (!this.state.dashboardData?.kpis) return;

        // Executive Summary KPIs
        this.updateKPISection('financial-kpis', [
            { title: 'Replacement Cost', value: `${this.state.dashboardData.kpis.replacementCost}K`, class: 'danger' },
            { title: 'Recruiting Spend', value: `${this.state.dashboardData.kpis.recruitingSpend}K`, class: 'warning' },
            { title: 'Revenue at Risk', value: `${this.state.dashboardData.kpis.revenueAtRisk}K`, class: 'info' },
            { title: 'Current Attrition', value: `${this.state.dashboardData.kpis.currentAttrition}%`, class: 'success' }
        ]);

        // Daily Pulse KPIs
        this.updateKPISection('daily-pulse-kpis', [
            { title: 'New Hires (MTD)', value: this.state.dashboardData.kpis.newHires, class: 'primary' },
            { title: 'Open Positions', value: this.state.dashboardData.kpis.openPositions, class: 'warning' },
            { title: 'Interviews Today', value: this.state.dashboardData.kpis.interviewsToday, class: 'info' },
            { title: 'Exit Interviews', value: this.state.dashboardData.kpis.exitInterviews, class: 'danger' }
        ]);
    }

    updateKPISection(sectionId, kpis) {
        const section = document.getElementById(sectionId);
        if (!section) return;

        const kpiHTML = kpis.map(kpi => `
            <div class="col-md-3 col-6 mb-3">
                <div class="kpi-card ${kpi.class}">
                    <div class="kpi-value">${kpi.value}</div>
                    <div class="kpi-label">${kpi.title}</div>
                </div>
            </div>
        `).join('');

        section.innerHTML = kpiHTML;
    }

    redrawCharts() {
        // Redraw all Plotly charts to handle responsive resizing
        const chartContainers = document.querySelectorAll('.chart-container');
        chartContainers.forEach(container => {
            if (container.children.length > 0) {
                Plotly.Plots.resize(container.firstElementChild);
            }
        });
    }

    updateTimestamp() {
        this.state.lastUpdated = new Date();
        const timeElement = document.getElementById('last-updated-time');
        if (timeElement) {
            timeElement.textContent = this.formatTimestamp(this.state.lastUpdated);
        }
    }

    formatTimestamp(date) {
        return new Intl.DateTimeFormat('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        }).format(date);
    }

    startRealTimeUpdates() {
        if (!this.state.realTimeEnabled) return;

        setInterval(() => {
            this.updateTimestamp();
            // Update only real-time sections
            this.updateRealTimeData();
        }, this.state.refreshInterval);
    }

    updateRealTimeData() {
        // Update KPIs with slight variations
        if (this.state.dashboardData?.kpis) {
            const kpis = this.state.dashboardData.kpis;
            kpis.newHires = Math.max(0, kpis.newHires + Math.floor(Math.random() * 3 - 1));
            kpis.interviewsToday = Math.max(0, kpis.interviewsToday + Math.floor(Math.random() * 5 - 2));
            kpis.exitInterviews = Math.max(0, kpis.exitInterviews + Math.floor(Math.random() * 2 - 1));
            
            this.updateKPIs();
        }
    }

    async refreshDashboard() {
        this.showNotification('Refreshing dashboard...', 'info');
        try {
            await this.loadDashboardData();
            this.showNotification('Dashboard refreshed successfully', 'success');
        } catch (error) {
            this.showNotification('Failed to refresh dashboard', 'error');
        }
    }

    showNotification(message, type = 'info') {
        // Create toast notification
        const toastHTML = `
            <div class="toast align-items-center text-white bg-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'primary'} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">${message}</div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        let toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.ctrlKey || e.metaKey) {
                    // Add shortcut handling logic here if needed
                }
            });
    console.log(`Exporting dashboard as ${format}`);
    // Implement export functionality
    if (window.hrDashboard) {
        window.hrDashboard.showNotification(`Exporting dashboard as ${format.toUpperCase()}...`, 'info');
    }
};

window.alertHighRisk = () => {
    if (window.hrDashboard) {
        window.hrDashboard.showNotification('High-risk employees have been flagged for immediate attention', 'warning');
    }
};

window.openPlanningModal = () => {
    console.log('Opening workforce planning modal');
    // This would open a modal for scenario planning
};

// Initialize dashboard when script loads
document.addEventListener('DOMContentLoaded', () => {
    window.hrDashboard = new HRDashboard();
});

    renderCompensationIntelligence() {
        const data = this.state.dashboardData.charts.compensationIntelligence;
        const layout = { 
            ...this.chartConfigs.theme.layout, 
            title: '',
            xaxis: { title: 'Average Salary' },
            yaxis: { title: 'Risk Score' }
        };
        return Plotly.newPlot('compensation-intelligence-chart', data, layout, this.chartConfigs);
    }

    renderLearningROI() {
        const data = [this.state.dashboardData.charts.learningROI];
        const layout = { 
            ...this.chartConfigs.theme.layout, 
            title: '',
            xaxis: { title: 'Training Hours' },
            yaxis: { title: 'Score' }
        };
        return Plotly.newPlot('learning-roi-chart', data, layout, this.chartConfigs);
    }

    renderManagerPerformance() {
        const data = this.state.dashboardData.charts.managerPerformance;
        const layout = { 
            ...this.chartConfigs.theme.layout, 
            title: '',
            xaxis: { title: 'Manager Satisfaction' },
            yaxis: { title: 'Team Risk Score' }
        };
        return Plotly.newPlot('manager-performance-chart', data, layout, this.chartConfigs);
    }

    renderTalentPipeline() {
        const data = [this.state.dashboardData.charts.talentPipeline];
        const layout = { ...this.chartConfigs.theme.layout, title: '' };
        return Plotly.newPlot('talent-pipeline-chart', data, layout, this.chartConfigs);
    }

    renderJourneyMapping() {
        const data = [this.state.dashboardData.charts.journeyMapping];
        const layout = { 
            ...this.chartConfigs.theme.layout, 
            title: '',
            xaxis: { title: 'Tenure' },
            yaxis: { title: 'Job Satisfaction' }
        };
        return Plotly.newPlot('journey-mapping-chart', data, layout, this.chartConfigs);
    }

    renderCompensationAnalytics() {
        const data = [this.state.dashboardData.charts.compensationAnalytics];
        const layout = { 
            ...this.chartConfigs.theme.layout, 
            title: '',
            xaxis: { title: 'Job Role', tickangle: -45 },
            yaxis: { title: 'Average Salary' }
        };
        return Plotly.newPlot('compensation-analytics-chart', data, layout, this.chartConfigs);
    }
}