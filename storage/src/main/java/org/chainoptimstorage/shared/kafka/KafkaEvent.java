package org.chainoptimstorage.shared.kafka;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.chainoptimstorage.shared.enums.Feature;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class KafkaEvent<T> {

    private T newEntity;
    private T oldEntity;
    private Feature entityType;
    private EventType eventType;
    private Integer mainEntityId;
    private Feature mainEntityType;
    private String mainEntityName;

    public enum EventType {
        CREATE, UPDATE, DELETE
    }
}
