package org.chainoptimnotifications.internal.settings.service;

import org.chainoptimnotifications.internal.settings.model.UserSettings;

import java.util.List;

public interface UserSettingsRepository {

    List<UserSettings> getSettingsByUserIds(List<String> userIds);
}
