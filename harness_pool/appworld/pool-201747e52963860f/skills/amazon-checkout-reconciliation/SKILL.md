---
name: amazon-checkout-reconciliation
description: "Use only when the primary requested action is to place an Amazon order or modify an Amazon cart. It guides clean-cart reconciliation, recipient and address resolution, constraint and inventory validation, discount selection, checkout, and order verification. Do not use it when Amazon is only a data source for another app or for read-only, order-history, payment-history, or return requests."
whenToUse: "Use only when the user asks you to buy, order, or change an Amazon cart, or apply a discount before checkout. Do not use it for read-only cart, wishlist, order-history, payment-history, return, or financial-reconciliation requests, and never clear a cart merely because it is nonempty."
disable-model-invocation: false
user-invocable: false
---

After loading this Skill, use the environment's execute tool for every app operation; never call an app API, Python snippet, or Skill name directly as a tool name.

Before any Amazon write, derive the exact requested product set, quantities, variations, constraints, recipient, destination, payment preference, and discount source from the request and any already retrieved source messages. If a required fact is missing, read the relevant documented app data before changing the cart. Resolve the recipient and delivery address from authoritative profile or contact data before checkout; if the requested address is not already saved in Amazon, add or select the matching address and never substitute an unrelated saved address.

Inspect the current cart before adding or removing anything. If the request is to buy the entire current cart, preserve its items and applied discount. Otherwise, remove or clear unrelated entries and rebuild the cart so it contains only the requested products. Represent each requested product as one cart entry with its exact total quantity; preserve requested gift wrapping and delivery constraints.

Select candidates that satisfy every stated product, seller, price, rating, review, inventory, delivery, color, size, and availability constraint. Follow the stated preference order. If a candidate is unavailable or fails a required constraint, move to the next valid candidate before spending remaining execution budget. Batch related documented reads and safe writes inside one environment execute call when possible, but do not repeat an identical failed call.

For discount selection, test each candidate independently and catch invalid-code errors so one failure does not stop the remaining candidates. Compare valid discounts on the same requested cart, then restore the best valid option before checkout. Do not spend the remaining budget searching for undocumented discount lists.

Before ordering, verify the cart contents, quantities, constraints, discount state, delivery destination, and a non-expired payment card. If the request needs multiple destinations, gift-wrap variants, or separate orders, rebuild and verify the cart for each order. Place an order only after the cart is clean, the recipient and address are resolved, and every constraint is satisfied.

After each order, read the resulting order and verify its products, quantities, discount, address, payment, and delivery requirements. Complete only after every required order is verified; for a side-effect request, finish without a natural-language answer.
