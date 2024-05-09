package org.chainoptimnotifications.core.notification.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.chainoptimnotifications.shared.enums.Feature;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UpdateNotificationDTO {

    private Integer id;
    private String userId;
    private String title;
    private Integer entityId;
    private Feature entityType;
    private String message;
    private Boolean readStatus;
    private String type;
}
