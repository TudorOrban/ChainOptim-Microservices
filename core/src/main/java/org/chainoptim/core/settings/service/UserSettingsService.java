package org.chainoptim.core.settings.service;

import org.chainoptim.core.settings.dto.CreateUserSettingsDTO;
import org.chainoptim.core.settings.dto.UpdateUserSettingsDTO;
import org.chainoptim.core.settings.model.UserSettings;

import java.util.List;

public interface UserSettingsService {

    UserSettings getUserSettings(String userId);
    List<UserSettings> getSettingsByUserIdIn(List<String> userIds);

    UserSettings saveUserSettings(CreateUserSettingsDTO userSettingsDTO);

    UserSettings updateUserSettings(UpdateUserSettingsDTO userSettingsDTO);

    void deleteUserSettings(Integer id);
}
