package org.chainoptimnotifications.config;


import org.chainoptimnotifications.outsidefeatures.model.ClientOrderEvent;
import org.chainoptimnotifications.outsidefeatures.model.SupplierOrderEvent;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.chainoptimnotifications.outsidefeatures.model.UserEvent;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory;
import org.springframework.kafka.core.ConsumerFactory;
import org.springframework.kafka.core.DefaultKafkaConsumerFactory;
import org.springframework.kafka.support.serializer.JsonDeserializer;

import java.util.HashMap;
import java.util.Map;
@Configuration
public class KafkaConsumerConfig {

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, SupplierOrderEvent> supplierOrderKafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, SupplierOrderEvent> factory = new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(supplierOrderEventConsumerFactory());
        return factory;
    }

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, ClientOrderEvent> clientOrderKafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, ClientOrderEvent> factory = new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(clientOrderEventConsumerFactory());
        return factory;
    }

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, UserEvent> userKafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, UserEvent> factory = new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(userEventConsumerFactory());
        return factory;
    }

    @Bean
    public ConsumerFactory<String, SupplierOrderEvent> supplierOrderEventConsumerFactory() {
        Map<String, Object> props = new HashMap<>();
        configureCommonConsumerProperties(props);

        JsonDeserializer<SupplierOrderEvent> valueDeserializer = new JsonDeserializer<>(SupplierOrderEvent.class, false);
        valueDeserializer.ignoreTypeHeaders();
        valueDeserializer.addTrustedPackages("*");

        return new DefaultKafkaConsumerFactory<>(
                props,
                new StringDeserializer(),
                valueDeserializer
        );
    }

    @Bean
    public ConsumerFactory<String, ClientOrderEvent> clientOrderEventConsumerFactory() {
        Map<String, Object> props = new HashMap<>();
        configureCommonConsumerProperties(props);

        JsonDeserializer<ClientOrderEvent> valueDeserializer = new JsonDeserializer<>(ClientOrderEvent.class, false);
        valueDeserializer.ignoreTypeHeaders();
        valueDeserializer.addTrustedPackages("*");

        return new DefaultKafkaConsumerFactory<>(
                props,
                new StringDeserializer(),
                valueDeserializer
        );
    }

    @Bean
    public ConsumerFactory<String, UserEvent> userEventConsumerFactory() {
        Map<String, Object> props = new HashMap<>();
        configureCommonConsumerProperties(props);

        JsonDeserializer<UserEvent> valueDeserializer = new JsonDeserializer<>(UserEvent.class, false);
        valueDeserializer.ignoreTypeHeaders();
        valueDeserializer.addTrustedPackages("*");

        return new DefaultKafkaConsumerFactory<>(
                props,
                new StringDeserializer(),
                valueDeserializer
        );
    }

    private void configureCommonConsumerProperties(Map<String, Object> props) {
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "kafka:9092");
        props.put(ConsumerConfig.GROUP_ID_CONFIG, "notification-group");
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, JsonDeserializer.class.getName());
    }
}
