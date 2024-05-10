package org.chainoptim.internalcommunication.out.controller;

import org.chainoptim.core.settings.model.UserSettings;
import org.chainoptim.core.settings.service.UserSettingsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/internal/settings")
public class SettingsInternalController {

    private final UserSettingsService userSettingsService;

    @Autowired
    public SettingsInternalController(UserSettingsService userSettingsService) {
        this.userSettingsService = userSettingsService;
    }

    @GetMapping("/users")
    public ResponseEntity<List<UserSettings>> getSettingsByUserIds(@RequestParam List<String> userIds) {
        List<UserSettings> userSettings = userSettingsService.getSettingsByUserIdIn(userIds);
        if (userSettings != null) {
            return ResponseEntity.ok(userSettings);
        } else {
            return ResponseEntity.notFound().build();
        }
    }
}