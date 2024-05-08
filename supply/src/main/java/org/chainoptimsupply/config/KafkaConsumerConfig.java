package org.chainoptimsupply.config;

import org.chainoptimsupply.kafka.OrganizationEvent;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
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
    public ConcurrentKafkaListenerContainerFactory<String, OrganizationEvent> organizationKafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, OrganizationEvent> factory = new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(organizationEventConsumerFactory());
        return factory;
    }

    @Bean
    public ConsumerFactory<String, OrganizationEvent> organizationEventConsumerFactory() {
        Map<String, Object> props = new HashMap<>();
        configureCommonConsumerProperties(props);

        JsonDeserializer<OrganizationEvent> valueDeserializer = new JsonDeserializer<>(OrganizationEvent.class, false);
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
        props.put(ConsumerConfig.GROUP_ID_CONFIG, "supply-organization-group");
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, JsonDeserializer.class.getName());
    }
}
