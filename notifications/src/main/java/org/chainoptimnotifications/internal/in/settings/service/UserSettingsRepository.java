package org.chainoptimnotifications.internal.in.settings.service;

import org.chainoptimnotifications.internal.in.settings.model.UserSettings;

import java.util.List;

public interface UserSettingsRepository {

    List<UserSettings> getSettingsByUserIds(List<String> userIds);
}
