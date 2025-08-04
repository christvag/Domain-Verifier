#!/usr/bin/env python3
"""
Contact Link Classifier Trainer

Trains and saves a binary classifier for contact link detection.
Run this once to create the trained model, then use it in the crawler.

Usage:
    python contact_link_trainer.py
"""

import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import numpy as np

# Suppress transformer deprecation warnings
warnings.filterwarnings(
    "ignore", message=".*encoder_attention_mask.*", category=FutureWarning
)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Missing required dependencies!")
    print("pip install sentence-transformers scikit-learn")
    exit(1)


class ContactLinkTrainer:
    """Train a binary classifier for contact link detection"""

    def __init__(self):
        self.sentence_model = None
        self.classifier = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

    def augment_text_with_synonyms(self, text):
        """Apply synonym augmentation to text"""
        synonyms = {
            "contact": ["reach", "get in touch", "connect", "message", "write", "call"],
            "about": ["company", "story", "team", "who we are", "our story"],
            "support": ["help", "assistance", "aid", "service", "customer service"],
            "help": ["support", "assistance", "aid", "guidance", "service"],
            "sales": ["business", "inquiries", "quotes", "pricing", "commercial"],
            "inquiry": ["question", "request", "message", "contact", "query"],
            "quote": ["estimate", "pricing", "cost", "price", "bid"],
            "appointment": [
                "booking",
                "schedule",
                "meeting",
                "session",
                "consultation",
            ],
            "service": ["support", "help", "assistance", "aid", "maintenance"],
            "information": ["details", "info", "data", "facts", "contact"],
            "form": ["contact form", "inquiry form", "request form", "message form"],
            "page": ["section", "area", "link", "page", "section"],
            "center": ["hub", "section", "area", "department", "division"],
            "team": ["staff", "crew", "members", "people", "personnel"],
            "company": [
                "business",
                "organization",
                "firm",
                "enterprise",
                "corporation",
            ],
            "customer": ["client", "user", "patron", "buyer", "consumer"],
            "business": [
                "commercial",
                "corporate",
                "enterprise",
                "professional",
                "company",
            ],
            "professional": [
                "expert",
                "specialist",
                "qualified",
                "certified",
                "licensed",
            ],
            "service": [
                "maintenance",
                "repair",
                "installation",
                "support",
                "assistance",
            ],
            "quality": ["standard", "grade", "level", "caliber", "excellence"],
        }

        augmented_texts = [text]

        # Apply synonym replacement
        for word, syns in synonyms.items():
            if word.lower() in text.lower():
                for syn in syns[:2]:  # Limit to 2 synonyms per word
                    new_text = text.lower().replace(word.lower(), syn.lower())
                    if new_text != text.lower():
                        augmented_texts.append(new_text)

        # Apply case variations
        if len(augmented_texts) < 3:
            augmented_texts.append(text.upper())
            augmented_texts.append(text.title())

        # Apply spacing variations
        if len(augmented_texts) < 4:
            augmented_texts.append(text.replace(" ", ""))
            augmented_texts.append(text.replace(" ", "-"))

        return list(set(augmented_texts))  # Remove duplicates

    def get_training_data(self):
        """Get comprehensive training data for contact link classification"""

        contact_examples = [
            # Direct contact - simple and clear
            "Contact Us",
            "Contact",
            "Get in Touch",
            "Contact Information",
            "Contact Details",
            "Contact Form",
            "Contact Page",
            "Reach Out",
            "Reach Us",
            # Support - common variations
            "Support",
            "Help",
            "Customer Support",
            "Customer Service",
            "Help Center",
            "Help Desk",
            "Get Help",
            "Need Help",
            "Support Center",
            "Technical Support",
            # About pages - often have contact info
            "About Us",
            "About",
            "About Our Company",
            "Company Information",
            "Our Story",
            "Meet The Team",
            "About the Company",
            "Who We Are",
            # Sales related
            "Sales",
            "Sales Team",
            "Business Inquiries",
            "Get Quote",
            "Request Quote",
            "Sales Contact",
            "Sales Support",
            "Business Development",
            # Feedback/Communication
            "Feedback",
            "Questions",
            "Ask Us",
            "Send Feedback",
            "Inquiry",
            "General Inquiries",
            "Leave Message",
            "Submit Question",
            # Communication methods
            "Call Us",
            "Phone Us",
            "Email Us",
            "Message Us",
            "Write Us",
            "Live Chat",
            "Chat Support",
            "Online Support",
            # URL path context examples (what we actually see)
            "Contact Us /contact",
            "About Us /about",
            "Contact /contact-us",
            "Support /support",
            "Help /help-center",
            "Contact Information /contact-info",
            "Get in Touch /contact",
            "Customer Support /support",
            "About /about-us",
            "Contact Form /contact",
            "Reach Us /contact",
            "Help Center /help",
            "Contact Details /contact",
            "Support Center /support",
            "Contact Page /contact",
            "About Our Company /about",
            "Customer Service /support",
            "Get Quote /quote",
            "Request Quote /quote",
            "Sales Contact /sales",
            "Business Inquiries /contact",
            "Feedback /feedback",
            "Ask Us /contact",
            "Inquiry /contact",
            "General Inquiries /contact",
            "Call Us /contact",
            "Email Us /contact",
            "Message Us /contact",
            "Live Chat /chat",
            "Contact Us Home|About Us|Contact Us|My Account",
            "About Us Home|About Us|Contact Us|Privacy Policy",
            "Customer Support Help Center|Support|Documentation",
            "Help Center Support|Help|FAQ|Contact",
            "Sales Contact Sales|Products|Services|Contact",
            "Technical Support Support|Technical|Help|Contact",
            "Contact Information Footer|Contact|Terms|Privacy",
            "Support Center Navigation|Support|Help|Resources",
            "Get in Touch Contact|Information|Details|Form",
        ]

        non_contact_examples = [
            # Product/Service pages
            "Products",
            "Services",
            "Solutions",
            "Features",
            "Browse Products",
            "Product Details",
            "Service Details",
            "View Products",
            "Our Products",
            "Product Catalog",
            "Service Offerings",
            "What We Do",
            "Our Services",
            # HVAC Specific Services (common misclassifications)
            "Heating",
            "Air Conditioning",
            "Air Quality",
            "Water Heaters",
            "Commercial Services",
            "HVAC Services",
            "Heating Services",
            "Cooling Services",
            "Air Conditioning Services",
            "Heating Installation",
            "AC Installation",
            "Heating Repair",
            "AC Repair",
            "Heating Maintenance",
            "AC Maintenance",
            "Heating Systems",
            "Cooling Systems",
            "HVAC Systems",
            "Heating Equipment",
            "Cooling Equipment",
            "HVAC Equipment",
            "Heating Solutions",
            "Cooling Solutions",
            "HVAC Solutions",
            "Heating Products",
            "Cooling Products",
            "HVAC Products",
            # E-commerce
            "Shop",
            "Store",
            "Buy Now",
            "Purchase",
            "Add to Cart",
            "Shopping Cart",
            "Cart",
            "Checkout",
            "Order Now",
            "Shop Now",
            "Categories",
            "Browse Store",
            "Product Search",
            "Departments",
            "Collections",
            "Buy Online",
            # Account/Auth
            "Login",
            "Sign In",
            "Register",
            "Sign Up",
            "My Account",
            "User Account",
            "Dashboard",
            "Settings",
            "Profile",
            "Account Settings",
            "Preferences",
            "Log Out",
            "Sign Out",
            "Create Account",
            # Content/Information
            "News",
            "Blog",
            "Articles",
            "Resources",
            "Documentation",
            "Guides",
            "Tutorials",
            "FAQ",
            "Knowledge Base",
            "Help Articles",
            "Support Docs",
            "User Guide",
            "Manual",
            "Instructions",
            # Navigation/Pages
            "Home",
            "Homepage",
            "Main Page",
            "Index",
            "Welcome",
            "Landing Page",
            "Overview",
            "Introduction",
            "Getting Started",
            "Quick Start",
            # Legal/Policy
            "Privacy Policy",
            "Terms of Service",
            "Terms and Conditions",
            "Legal",
            "Disclaimer",
            "Cookie Policy",
            "Privacy",
            "Terms",
            "Conditions",
            "Agreement",
            # URL path context for non-contact
            "Products /products",
            "Services /services",
            "Shop /shop",
            "Store /store",
            "Login /login",
            "Sign In /signin",
            "Register /register",
            "My Account /account",
            "Home /home",
            "Products /catalog",
            "Services /what-we-do",
            "Shop /buy",
            "Store /store",
            "Login /signin",
            "Register /signup",
            "My Account /profile",
            "Home /index",
            "Products /catalog/products",
            "Services /services/overview",
            "Shop /store/shop",
            "Store /store/browse",
            "Login /auth/login",
            "Sign In /auth/signin",
            "Register /auth/register",
            "My Account /user/account",
            "Home /index.html",
            "Products /catalog/index",
            "Services /services/list",
            "Shop /store/cart",
            "Store /store/checkout",
            "Login /user/login",
            "Sign In /user/signin",
            "Register /user/register",
            "My Account /user/profile",
            "Home /index.php",
            "Products /catalog/products.php",
            "Services /services/index.php",
            "Shop /store/shop.php",
            "Store /store/browse.php",
            "Login /auth/login.php",
            "Sign In /auth/signin.php",
            "Register /auth/register.php",
            "My Account /user/account.php",
            "Home /index.html",
            "Products /catalog/index.html",
            "Services /services/list.html",
            "Shop /store/cart.html",
            "Store /store/checkout.html",
            "Login /user/login.html",
            "Sign In /user/signin.html",
            "Register /user/register.html",
            "My Account /user/profile.html",
            # HVAC Service URLs (critical for distinction)
            "Heating /heating",
            "Air Conditioning /air-conditioning",
            "Air Quality /air-quality",
            "Water Heaters /water-heaters",
            "Commercial Services /commercial-services",
            "HVAC Services /hvac-services",
            "Heating Services /heating-services",
            "Cooling Services /cooling-services",
            "Air Conditioning Services /ac-services",
            "Heating Installation /heating-installation",
            "AC Installation /ac-installation",
            "Heating Repair /heating-repair",
            "AC Repair /ac-repair",
            "Heating Maintenance /heating-maintenance",
            "AC Maintenance /ac-maintenance",
            "Heating Systems /heating-systems",
            "Cooling Systems /cooling-systems",
            "HVAC Systems /hvac-systems",
            "Heating Equipment /heating-equipment",
            "Cooling Equipment /cooling-equipment",
            "HVAC Equipment /hvac-equipment",
            "Heating Solutions /heating-solutions",
            "Cooling Solutions /cooling-solutions",
            "HVAC Solutions /hvac-solutions",
            "Heating Products /heating-products",
            "Cooling Products /cooling-products",
            "HVAC Products /hvac-products",
        ]

        # Apply data augmentation to contact examples
        augmented_contact_examples = []
        for example in contact_examples:
            augmented_contact_examples.extend(self.augment_text_with_synonyms(example))

        # Apply data augmentation to non-contact examples
        augmented_non_contact_examples = []
        for example in non_contact_examples:
            augmented_non_contact_examples.extend(
                self.augment_text_with_synonyms(example)
            )

        return augmented_contact_examples, augmented_non_contact_examples

    def find_boundary_samples(self, embeddings, labels, n_samples=50):
        """Find samples near the decision boundary for focused training"""
        from sklearn.metrics.pairwise import cosine_similarity

        # Get current model predictions
        y_proba = self.classifier.predict_proba(embeddings)[:, 1]

        # Find samples near the decision boundary (0.5 threshold)
        boundary_distance = np.abs(y_proba - 0.5)
        boundary_indices = np.argsort(boundary_distance)[:n_samples]

        # Also include some clear misclassifications
        y_pred = self.classifier.predict(embeddings)
        misclassified = np.where(y_pred != labels)[0]

        # Combine boundary samples with misclassified samples
        critical_indices = np.union1d(boundary_indices, misclassified[:20])

        return critical_indices

    def train_with_vector_search_optimization(self):
        """Train the model using vector search to focus on critical samples"""
        print("🔄 Training with balanced vector search optimization...")

        # Get training data
        contact_examples, non_contact_examples = self.get_training_data()
        contact_labels = [1] * len(contact_examples)
        non_contact_labels = [0] * len(non_contact_examples)
        all_examples = contact_examples + non_contact_examples
        all_labels = contact_labels + non_contact_labels

        # Generate embeddings
        embeddings = self.sentence_model.encode(all_examples)

        # Initial training with full dataset
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        # Train initial model with full dataset
        self.classifier.fit(X_train, y_train)

        # Find critical samples near decision boundary
        critical_indices = self.find_boundary_samples(embeddings, all_labels)

        # Get predictions for analysis
        y_pred = self.classifier.predict(embeddings)
        misclassified_count = len(np.where(y_pred != all_labels)[0])

        print(f"📊 Critical samples found: {len(critical_indices)}")
        print(f"   - Near boundary: {len(critical_indices) - misclassified_count}")
        print(f"   - Misclassified: {misclassified_count}")

        # Create balanced training set: 70% full dataset + 30% critical samples
        X_critical = embeddings[critical_indices]
        y_critical = np.array(all_labels)[critical_indices]

        # Combine datasets with weights
        X_combined = np.vstack([X_train, X_critical])
        y_combined = np.concatenate([y_train, y_critical])

        # Use sample weights to give more importance to critical samples
        sample_weights = np.ones(len(X_combined))
        sample_weights[len(X_train) :] = 2.0  # Give 2x weight to critical samples

        # Retrain with balanced approach
        self.classifier.fit(X_combined, y_combined, sample_weight=sample_weights)

        return X_test, y_test

    def train_model(self):
        """Train the binary classifier with advanced techniques"""
        print("🤖 Training Contact Link Binary Classifier")
        print("=" * 50)

        # Load sentence transformer
        print("📥 Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Sentence transformer loaded")

        # Get training data with augmentation
        contact_examples, non_contact_examples = self.get_training_data()

        # Create labels
        contact_labels = [1] * len(contact_examples)
        non_contact_labels = [0] * len(non_contact_examples)

        all_examples = contact_examples + non_contact_examples
        all_labels = contact_labels + non_contact_labels

        print("📊 Training data:")
        print(f"   - Contact examples: {len(contact_examples)}")
        print(f"   - Non-contact examples: {len(non_contact_examples)}")
        print(f"   - Total examples: {len(all_examples)}")

        # Get embeddings
        print("🔄 Generating embeddings...")
        embeddings = self.sentence_model.encode(all_examples, show_progress_bar=True)

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        print("📈 Training classifier with vector search optimization...")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Test samples: {len(X_test)}")

        # Use advanced LogisticRegression with optimized parameters
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=2000,  # Increased iterations
            C=1.0,  # Regularization strength
            class_weight="balanced",  # Handle class imbalance
            solver="liblinear",  # Better for binary classification
            penalty="l2",  # L2 regularization
        )

        # Train with vector search optimization
        X_test, y_test = self.train_with_vector_search_optimization()

        # Evaluate with cross-validation
        from sklearn.model_selection import cross_val_score

        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5)

        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("\n📊 Model Performance:")
        print(f"   - Test Accuracy: {accuracy:.3f}")
        print(f"   - Cross-validation scores: {cv_scores}")
        print(f"   - CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        print("\n📋 Detailed Report:")
        print(
            classification_report(
                y_test, y_pred, target_names=["Non-Contact", "Contact"]
            )
        )

        # Feature importance analysis
        feature_importance = abs(self.classifier.coef_[0])
        print(f"\n🔍 Model Analysis:")
        print(
            f"   - Feature importance range: {feature_importance.min():.3f} - {feature_importance.max():.3f}"
        )
        print(f"   - Model confidence: {self.classifier.score(X_test, y_test):.3f}")

        return accuracy

    def find_optimal_threshold(self, X_test, y_test):
        """Find optimal threshold for classification"""
        from sklearn.metrics import precision_recall_curve, f1_score

        y_proba = self.classifier.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

        # Find threshold that maximizes F1 score
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1_scores.append(f1_score(y_test, y_pred))

        optimal_threshold = thresholds[np.argmax(f1_scores)]
        optimal_f1 = max(f1_scores)

        print(f"\n🎯 Optimal Threshold Analysis:")
        print(f"   - Optimal threshold: {optimal_threshold:.3f}")
        print(f"   - Optimal F1 score: {optimal_f1:.3f}")

        return optimal_threshold

    def save_models(self):
        """Save the trained models"""
        print(f"\n💾 Saving models to {self.model_dir}/")

        # Save sentence transformer
        sentence_model_path = self.model_dir / "sentence_model"
        self.sentence_model.save(str(sentence_model_path))
        print(f"   ✅ Sentence model saved to {sentence_model_path}")

        # Save classifier
        classifier_path = self.model_dir / "contact_classifier.pkl"
        with open(classifier_path, "wb") as f:
            pickle.dump(self.classifier, f)
        print(f"   ✅ Classifier saved to {classifier_path}")

        # Find optimal threshold
        from sklearn.model_selection import train_test_split

        contact_examples, non_contact_examples = self.get_training_data()
        contact_labels = [1] * len(contact_examples)
        non_contact_labels = [0] * len(non_contact_examples)
        all_examples = contact_examples + non_contact_examples
        all_labels = contact_labels + non_contact_labels
        embeddings = self.sentence_model.encode(all_examples)
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        optimal_threshold = self.find_optimal_threshold(X_test, y_test)

        # Save metadata with optimal threshold
        metadata = {
            "model_type": "contact_link_binary_classifier",
            "sentence_model": "all-MiniLM-L6-v2",
            "classifier": "LogisticRegression",
            "training_date": "unknown",
            "threshold": optimal_threshold,  # Use optimal threshold
            "training_techniques": [
                "synonym_augmentation",
                "cross_validation",
                "class_weight_balanced",
                "l2_regularization",
                "optimal_threshold_tuning",
            ],
            "augmentation_methods": [
                "synonym_replacement",
                "case_variations",
                "spacing_variations",
            ],
        }

        metadata_path = self.model_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            import json

            json.dump(metadata, f, indent=2)
        print(f"   ✅ Metadata saved to {metadata_path}")

    def test_predictions(self):
        """Test the model with some examples"""
        print("\n🧪 Testing model predictions:")

        test_examples = [
            "Contact Us",
            "About Us",
            "Support",
            "Help Center",
            "Products",
            "Shopping Cart",
            "Privacy Policy",
            "Tape Drives",
            "My Account",
        ]

        embeddings = self.sentence_model.encode(test_examples)
        predictions = self.classifier.predict_proba(embeddings)

        for example, prob in zip(test_examples, predictions):
            contact_prob = prob[1]
            prediction = "✅ CONTACT" if contact_prob > 0.3 else "❌ NON-CONTACT"
            print(f"   {prediction} ({contact_prob:.3f}) '{example}'")


def main():
    """Main training function"""
    trainer = ContactLinkTrainer()

    try:
        # Train the model
        accuracy = trainer.train_model()

        if accuracy > 0.75:  # Only save if accuracy is decent
            trainer.save_models()
            trainer.test_predictions()

            print("\n🎉 Training completed successfully!")
            print(f"   Model files saved in: {trainer.model_dir}/")
            print("   Ready to use in contact_form_crawler.py")
        else:
            print(
                f"\n⚠️ Model accuracy ({accuracy:.3f}) is too low. Consider improving training data."
            )

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
