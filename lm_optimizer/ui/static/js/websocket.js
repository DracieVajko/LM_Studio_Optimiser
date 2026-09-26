// LM Studio Auto Optimizer - WebSocket Manager

export class WebSocketManager {
    constructor() {
        this.connections = new Map();
        this.reconnectAttempts = new Map();
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
    }

    connect(runId, onMessage) {
        const ws = new WebSocket(`ws://${window.location.host}/ws/optimize/${runId}`);

        ws.onopen = () => {
            console.log(`WebSocket connected for run ${runId}`);
            this.reconnectAttempts.set(runId, 0);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(data);
            } catch (error) {
                console.error('WebSocket message parse error:', error);
            }
        };

        ws.onclose = () => {
            console.log(`WebSocket disconnected for run ${runId}`);
            this.attemptReconnect(runId, onMessage);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.connections.set(runId, ws);
        return ws;
    },

    disconnect(runId) {
        const ws = this.connections.get(runId);
        if (ws) {
            ws.close();
            this.connections.delete(runId);
            this.reconnectAttempts.delete(runId);
        }
    },

    disconnectAll() {
        for (const [runId, ws] of this.connections) {
            ws.close();
        }
        this.connections.clear();
        this.reconnectAttempts.clear();
    },

    attemptReconnect(runId, onMessage) {
        const attempts = this.reconnectAttempts.get(runId) || 0;

        if (attempts >= this.maxReconnectAttempts) {
            console.log(`Max reconnect attempts reached for run ${runId}`);
            return;
        }

        this.reconnectAttempts.set(runId, attempts + 1);
        const delay = this.reconnectDelay * Math.pow(1.5, attempts);

        console.log(`Reconnecting in ${delay}ms (attempt ${attempts + 1})`);

        setTimeout(() => {
            this.connect(runId, onMessage);
        }, delay);
    },

    send(runId, message) {
        const ws = this.connections.get(runId);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(message));
        }
    },

    isConnected(runId) {
        const ws = this.connections.get(runId);
        return ws && ws.readyState === WebSocket.OPEN;
    },
}