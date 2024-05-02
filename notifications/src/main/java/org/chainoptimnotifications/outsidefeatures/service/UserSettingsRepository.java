package org.chainoptimnotifications.outsidefeatures.service;

import org.chainoptimnotifications.outsidefeatures.model.UserSettings;

import java.util.List;

public interface UserSettingsRepository {

    List<UserSettings> getSettingsByUserIds(List<String> userIds);
}
