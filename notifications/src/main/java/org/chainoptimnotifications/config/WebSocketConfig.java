package org.chainoptimnotifications.config;

import org.chainoptimnotifications.notification.websocket.SimpleTextWebSocketHandler;
import org.chainoptimnotifications.notification.websocket.UserHandshakeInterceptor;
import org.chainoptimnotifications.notification.websocket.WebSocketMessagingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    private final WebSocketMessagingService messagingService;

    @Autowired
    public WebSocketConfig(WebSocketMessagingService messagingService) {
        this.messagingService = messagingService;
    }

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(simpleTextWebSocketHandler(messagingService), "/ws")
                .setAllowedOrigins("*")
                .addInterceptors(new UserHandshakeInterceptor());
    }

    @Bean
    public SimpleTextWebSocketHandler simpleTextWebSocketHandler(WebSocketMessagingService messagingService) {
        return new SimpleTextWebSocketHandler(messagingService);
    }
}
