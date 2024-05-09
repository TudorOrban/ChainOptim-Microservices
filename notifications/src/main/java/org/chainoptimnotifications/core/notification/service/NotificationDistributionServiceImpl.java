package org.chainoptimnotifications.core.notification.service;

import org.chainoptimnotifications.core.notification.model.KafkaEvent;
import org.chainoptimnotifications.core.notification.model.NotificationUserDistribution;
import org.chainoptimnotifications.shared.enums.Feature;
import org.chainoptimnotifications.internal.demand.model.ClientOrderEvent;
import org.chainoptimnotifications.internal.settings.model.NotificationSettings;
import org.chainoptimnotifications.internal.supply.model.SupplierOrderEvent;
import org.chainoptimnotifications.internal.settings.model.UserSettings;
import org.chainoptimnotifications.internal.service.UserSettingsRepository;
import org.chainoptimnotifications.shared.redis.RedisService;
import org.chainoptimnotifications.internal.tenant.model.FeaturePermissions;
import org.chainoptimnotifications.internal.tenant.model.Organization;
import org.chainoptimnotifications.internal.tenant.model.Permissions;
import org.chainoptimnotifications.internal.tenant.model.User;
import org.chainoptimnotifications.internal.tenant.service.CachedOrganizationRepository;
import org.chainoptimnotifications.internal.tenant.service.OrganizationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

@Service
public class NotificationDistributionServiceImpl implements NotificationDistributionService {

//    private final SupplierRepository supplierRepository;
//    private final ClientRepository clientRepository;
    private final OrganizationRepository organizationRepository;
    private final CachedOrganizationRepository cachedOrganizationRepository;
    private final UserSettingsRepository userSettingsRepository;
    private final RedisService redisService;

    @Autowired
    public NotificationDistributionServiceImpl(OrganizationRepository organizationRepository,
                                               CachedOrganizationRepository cachedOrganizationRepository,
                                               UserSettingsRepository userSettingsRepository,
                                               RedisService redisService) {
        this.organizationRepository = organizationRepository;
        this.cachedOrganizationRepository = cachedOrganizationRepository;
        this.userSettingsRepository = userSettingsRepository;
        this.redisService = redisService;
    }

    public NotificationUserDistribution distributeEventToUsers(SupplierOrderEvent event) {
        System.out.println("Distributing event in notifications ms: " + event);
//        Integer organizationId = determineOrderOrganization(event);
        Integer organizationId = 1;

        return distributeEventToUsers(organizationId, event.getEventType(), event.getEntityType());
    }

    public NotificationUserDistribution distributeEventToUsers(ClientOrderEvent event) {
        System.out.println("Distributing event in notifications ms: " + event);
//        Integer organizationId = determineOrderOrganization(event);
        Integer organizationId = 1;

        return distributeEventToUsers(organizationId, event.getEventType(), event.getEntityType());
    }

    private NotificationUserDistribution distributeEventToUsers(Integer organizationId, KafkaEvent.EventType eventType, Feature entityType) {
        // TODO: Cache this with Redis
        Organization organization;
        if (redisService.isRedisAvailable()) {
            organization = cachedOrganizationRepository.getOrganizationWithUsersAndCustomRoles(organizationId);
        } else {
            organization = organizationRepository.getOrganizationWithUsersAndCustomRoles(organizationId);
        }

        List<String> candidateUserIds = new ArrayList<>();
        List<User> emailCandidateUsers = new ArrayList<>();
        if (organization == null) {
            return new NotificationUserDistribution(candidateUserIds, candidateUserIds);
        }

        // Skip if subscription plan doesn't support notifications
        if (organization.getSubscriptionPlanDetails() != null && !organization.getSubscriptionPlanDetails().isRealTimeNotificationsOn()) {
            return new NotificationUserDistribution(candidateUserIds, candidateUserIds);
        }

        determineMembersWithPermissions(organization.getUsers(), eventType, entityType, candidateUserIds, emailCandidateUsers);

        System.out.println("Candidate user IDs: " + candidateUserIds);
        // Determine which candidate users should receive this event based on their settings
        List<UserSettings> userSettings = userSettingsRepository.getSettingsByUserIds(candidateUserIds);
        List<String> usersToReceiveNotification = candidateUserIds.stream().filter(userId -> {
            UserSettings settings = userSettings.stream().filter(userSetting -> userSetting.getUserId().equals(userId)).findFirst().orElse(null);
            return settings != null && shouldReceiveNotification(settings.getNotificationSettings(), entityType);
        }).toList();
        System.out.println("Users to receive notification: " + usersToReceiveNotification);

        List<String> usersToReceiveEmail = emailCandidateUsers.stream().filter(user -> {
            UserSettings settings = userSettings.stream().filter(userSetting -> userSetting.getUserId().equals(user.getId())).findFirst().orElse(null);
            return settings != null && shouldReceiveEmailNotification(settings.getNotificationSettings(), entityType);
        }).map(User::getEmail).toList();

        return new NotificationUserDistribution(usersToReceiveNotification, usersToReceiveEmail);
    }

    private void determineMembersWithPermissions(Set<User> organizationMembers,
                                                 KafkaEvent.EventType eventType, Feature entityType,
                                                 List<String> candidateUserIds, List<User> emailCandidateUsers) {
        for (User user : organizationMembers) {
            if (user.getCustomRole() == null) {
                if (user.getRole().equals(User.Role.ADMIN)) {
                    candidateUserIds.add(user.getId());
                    emailCandidateUsers.add(user);
                }
                continue;
            }

            FeaturePermissions featurePermissions = getFeaturePermissions(user.getCustomRole().getPermissions(), entityType);
            if (featurePermissions == null) continue;
            if (hasPermissions(featurePermissions, eventType)) {
                candidateUserIds.add(user.getId());
                emailCandidateUsers.add(user);
            }
        }
    }

//    private Integer determineOrderOrganization(SupplierOrderEvent event) {
//        return supplierRepository.findOrganizationIdById(Long.valueOf(event.getNewEntity().getSupplierId()))
//                .orElseThrow(() -> new ResourceNotFoundException("Supplier with ID: " + event.getNewEntity().getSupplierId() + " not found"));
//    }
//
//    private Integer determineOrderOrganization(ClientOrderEvent event) {
//        return clientRepository.findOrganizationIdById(Long.valueOf(event.getNewEntity().getClientId()))
//                .orElseThrow(() -> new ResourceNotFoundException("Client with ID: " + event.getNewEntity().getClientId() + " not found"));
//    }

    private FeaturePermissions getFeaturePermissions(Permissions permissions, Feature entityType) {
        if (permissions == null) return null;
        return switch (entityType) {
            case SUPPLIER_ORDER -> permissions.getSuppliers();
            case CLIENT_ORDER -> permissions.getClients();
            case FACTORY_INVENTORY -> permissions.getFactories();
            case WAREHOUSE_INVENTORY -> permissions.getWarehouses();
            default -> null;
        };
    }

    private boolean hasPermissions(FeaturePermissions featurePermissions, KafkaEvent.EventType eventType) {
        return switch (eventType) {
            case CREATE -> Boolean.TRUE.equals(featurePermissions.getCanCreate());
            case UPDATE -> Boolean.TRUE.equals(featurePermissions.getCanUpdate());
            case DELETE -> Boolean.TRUE.equals(featurePermissions.getCanDelete());
        };
    }

    private boolean shouldReceiveNotification(NotificationSettings notificationSettings, Feature entityType) {
        return switch (entityType) {
            case SUPPLIER_ORDER -> notificationSettings.isSupplierOrdersOn();
            case CLIENT_ORDER -> notificationSettings.isClientOrdersOn();
            case FACTORY_INVENTORY -> notificationSettings.isFactoryInventoryOn();
            case WAREHOUSE_INVENTORY -> notificationSettings.isWarehouseInventoryOn();
            default -> false;
        };
    }

    private boolean shouldReceiveEmailNotification(NotificationSettings notificationSettings, Feature entityType) {
        return switch (entityType) {
            case SUPPLIER_ORDER -> notificationSettings.isEmailSupplierOrdersOn();
            case CLIENT_ORDER -> notificationSettings.isEmailClientOrdersOn();
            case FACTORY_INVENTORY -> notificationSettings.isEmailFactoryInventoryOn();
            case WAREHOUSE_INVENTORY -> notificationSettings.isEmailWarehouseInventoryOn();
            default -> false;
        };
    }
}
