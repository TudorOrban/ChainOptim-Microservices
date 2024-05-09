package org.chainoptimnotifications.core.notification.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.chainoptimnotifications.shared.enums.Feature;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class AddNotificationDTO {

    private List<String> userIds;
    private String title;
    private Integer entityId;
    private Feature entityType;
    private String message;
    private Boolean readStatus;
    private String type;
}
