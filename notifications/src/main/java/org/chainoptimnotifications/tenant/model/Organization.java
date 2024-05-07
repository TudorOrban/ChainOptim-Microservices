package org.chainoptimnotifications.tenant.model;

import jakarta.persistence.*;
import lombok.*;
import org.chainoptimnotifications.subscriptionplan.PlanDetails;
import org.chainoptimnotifications.subscriptionplan.SubscriptionPlans;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Objects;
import java.util.Set;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Organization implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private Integer id;
    private String name;
    private String address;
    private String contactInfo;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Set<User> users;

    public enum SubscriptionPlanTier {
        NONE,
        BASIC,
        PRO
    }

    @Enumerated(EnumType.STRING)
    @Column(name = "subscription_plan", nullable = false)
    private SubscriptionPlanTier subscriptionPlanTier;

    public PlanDetails getSubscriptionPlanDetails() {
        return SubscriptionPlans.getPlans().get(subscriptionPlanTier);
    }

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Organization that = (Organization) o;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
