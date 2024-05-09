package org.chainoptimnotifications.internal.settings.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UserSettings {

    private Integer id;
    private String userId;
    private GeneralSettings generalSettings;
    private NotificationSettings notificationSettings;

}
